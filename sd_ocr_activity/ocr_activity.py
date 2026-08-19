import gc
import json
import logging
import importlib.util
import os
import platform
import re
import time
import urllib3

import cv2
import numpy as np
import psutil
import pyopencl as cl
import requests

from sd_ocr_activity.const import CERT_FILE

os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)

logger = logging.getLogger(__name__)


def get_current_gpu_usage() -> str:
    """Detect and return GPU usage percentage across NVIDIA / AMD / Intel.
    Returns formatted string like '45.2%' or 'N/A' if unavailable."""
    # 1. Try NVIDIA NVML (Fastest & most accurate for NVIDIA)
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count > 0:
            usages = []
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                usages.append(util.gpu)
            pynvml.nvmlShutdown()
            avg_usage = sum(usages) / len(usages)
            return f"{avg_usage:.1f}%"
    except Exception:
        pass

    # 2. Windows Performance Counter Fallback (Works for AMD, Intel & NVIDIA on Windows)
    if platform.system() == "Windows":
        try:
            import win32pdh
            # Query the aggregated GPU Engine utilization counter
            query = win32pdh.OpenQuery()
            path = r"\GPU Engine(*)\Utilization Percentage"
            counter = win32pdh.AddCounter(query, path)
            win32pdh.CollectQueryData(query)
            time.sleep(0.05)
            win32pdh.CollectQueryData(query)
            _, val_dict = win32pdh.GetFormattedCounterArray(counter, win32pdh.PDH_FMT_DOUBLE)
            win32pdh.CloseQuery(query)
            
            total_gpu_pct = sum(item[1] for item in val_dict if item[1] > 0)
            if total_gpu_pct > 0:
                return f"{total_gpu_pct:.1f}%"
        except Exception:
            pass

    return "N/A"


class ActiveWindowOCRText:
    def __init__(self, server_url, screenshot_id, image_path, warmup=False) -> None:
        super().__init__()
        self._reader_cache = None
        self.server_url = server_url
        self.screenshot_id = screenshot_id
        self.image_path = image_path

        if warmup:
            self._warmup()

    def _warmup(self):
        try:
            reader = self.get_cached_reader()
            warmup_img = np.ones((256, 256, 3), dtype=np.uint8) * 255
            cv2.putText(warmup_img, "Warmup", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 3)
            _ = reader(warmup_img)
            del warmup_img
        except Exception:
            raise

    def _get_gpu_names(self):
        names = []
        for platform_item in cl.get_platforms():
            for device in platform_item.get_devices(device_type=cl.device_type.GPU):
                if device.available:
                    names.append(device.name.lower())
        return names

    def use_directml(self) -> bool:
        """Detect if system has GPU for DirectML acceleration."""
        try:
            active_gpus = []
            try:
                active_gpus = self._get_gpu_names()
            except Exception:
                pass

            if not active_gpus:
                logger.warning("[OCRText] No GPU name detected")
                return False

            logger.info(f"[OCRText] Detected active GPU: {active_gpus}")

            def has_any(text: str, keywords: list[str]) -> bool:
                return any(k in text for k in keywords)

            def has_any_regex(text: str, patterns: list[str]) -> bool:
                return any(re.search(p, text) for p in patterns)

            for gpu_name in active_gpus:
                gpu_name = gpu_name.lower()

                # NVIDIA
                if has_any(gpu_name, ["nvidia", "geforce", "quadro", "rtx", "gtx"]):
                    logger.info("[OCRText] Detected NVIDIA - DirectML")
                    return True

                # Intel Arc
                if has_any(gpu_name, ["intel arc", "arc a"]):
                    logger.info("[OCRText] Intel Arc detected - DirectML")
                    return True

                if has_any(gpu_name, ["iris xe", "intel", "uhd", "hd graphics", "iris"]):
                    logger.info("[OCRText] Intel iGPU detected")
                    continue

                # AMD / ATI
                if has_any(gpu_name, ["amd", "radeon", "ati"]):
                    logger.info("[OCRText] Detected AMD")

                    # Deny Vega integrated
                    vega_integrated_ids = ["3", "6", "7", "8", "10", "11"]
                    if any(re.search(rf"vega\s*{vid}\b", gpu_name) for vid in vega_integrated_ids):
                        continue

                    # Deny Vega 10 discrete
                    if "radeon vii" not in gpu_name:
                        if (re.search(r"vega\s*56\b", gpu_name) or
                                re.search(r"vega\s*64\b", gpu_name) or
                                "vega frontier" in gpu_name or
                                "vega" in gpu_name):
                            continue

                    # Deny Polaris
                    polaris_patterns = [r"rx\s*460\b", r"rx\s*470\b", r"rx\s*480\b",
                                        r"rx\s*550\b", r"rx\s*560\b", r"rx\s*570\b",
                                        r"rx\s*580\b", r"rx\s*590\b"]
                    if has_any_regex(gpu_name, polaris_patterns):
                        continue

                    # Deny R9 / R7 / R5
                    legacy_patterns = [
                        r"r9\s*295", r"r9\s*290", r"r9\s*280", r"r9\s*270",
                        r"r7\s*370", r"r7\s*360", r"r7\s*350", r"r7\s*340",
                        r"r7\s*260", r"r7\s*250", r"r7\s*240",
                        r"r5\s*340", r"r5\s*330", r"r5\s*240", r"r5\s*230"
                    ]
                    if has_any_regex(gpu_name, legacy_patterns):
                        continue

                    if has_any(gpu_name, ["hd 4", "hd 5", "hd 6"]):
                        continue

                    if has_any(gpu_name, ["radeon pro wx", "radeon pro w5", "radeon pro w4", "firepro w", "firepro s"]):
                        continue

                    if has_any(gpu_name, [" a4-", " a6-", " a8-", " a10-", " a12-",
                                          "bristol ridge", "kaveri", "carrizo", "stoney ridge"]):
                        continue

                    if "radeon graphics" in gpu_name:
                        rdna_igpu_markers = [
                            "880m", "870m", "860m", "780m", "760m", "740m",
                            "680m", "660m", "650m", "630m", "610m"
                        ]
                        if not has_any(gpu_name, rdna_igpu_markers):
                            continue

                    logger.info("[OCRText] Detected compatible AMD GPU")
                    return True

            return False

        except Exception:
            logger.exception("[OCRText] use_directml() failed")
            return False

    def has_intel_cpu(self) -> bool:
        """Rough check if CPU is Intel."""
        try:
            cpu_info = (platform.processor() or platform.machine() or "").lower()
            return "intel" in cpu_info
        except Exception:
            return False

    def get_cached_reader(self):
        if self._reader_cache is not None:
            return self._reader_cache

        try:
            from rapidocr import EngineType, OCRVersion, RapidOCR
        except Exception as e:
            raise RuntimeError(f"No suitable RapidOCR backend found. {e}")

        # GPU (DirectML)
        if self.use_directml():
            import onnxruntime as ort
            if "DmlExecutionProvider" in ort.get_available_providers():
                try:
                    self._reader_cache = RapidOCR(params={
                        "EngineConfig.onnxruntime.use_dml": True,
                        "Global.use_cls": False,
                        "Rec.ocr_version": OCRVersion.PPOCRV5,
                    })
                    return self._reader_cache
                except (requests.exceptions.RequestException, urllib3.exceptions.HTTPError, TimeoutError):
                    self._reader_cache = RapidOCR(params={"EngineConfig.onnxruntime.use_dml": True, "Global.use_cls": False})
                    return self._reader_cache
                except Exception as e:
                    logger.warning(f"[OCRText] DirectML failed to load: {e}")

        # Intel (OpenVINO)
        if self.has_intel_cpu() and importlib.util.find_spec("openvino") is not None:
            try:
                self._reader_cache = RapidOCR(params={
                    "Det.engine_type": EngineType.OPENVINO,
                    "Cls.engine_type": EngineType.OPENVINO,
                    "Rec.engine_type": EngineType.OPENVINO,
                    "Global.use_cls": False,
                    "Det.device_name": "AUTO",
                    "Cls.device_name": "AUTO",
                    "Rec.device_name": "AUTO",
                    "Rec.ocr_version": OCRVersion.PPOCRV5,
                })
                return self._reader_cache
            except (requests.exceptions.RequestException, urllib3.exceptions.HTTPError, TimeoutError):
                self._reader_cache = RapidOCR(params={
                    "Det.engine_type": EngineType.OPENVINO,
                    "Cls.engine_type": EngineType.OPENVINO,
                    "Rec.engine_type": EngineType.OPENVINO,
                    "Global.use_cls": False,
                    "Det.device_name": "AUTO",
                    "Cls.device_name": "AUTO",
                    "Rec.device_name": "AUTO",
                })
                return self._reader_cache
            except Exception as e:
                logger.warning(f"[OCRText] OpenVINO failed to load: {e}")

        # CPU Fallback
        try:
            self._reader_cache = RapidOCR(params={"Global.use_cls": False, "Rec.ocr_version": OCRVersion.PPOCRV5})
            return self._reader_cache
        except Exception:
            self._reader_cache = RapidOCR(params={"Global.use_cls": False})
            return self._reader_cache

    def save_30_percent_image(self, crop_img):
        tmp_file_path, ext = os.path.splitext(self.image_path)
        screenshot_file = f"{tmp_file_path}-30.png"
        cv2.imwrite(screenshot_file, crop_img)

    def _send_ocr_result(self, json_output):
        try:
            payload = {
                'screenshot_id': self.screenshot_id,
                'ocr_text': json.dumps(json_output)
            }
            with requests.post(self.server_url, json=payload, verify=str(CERT_FILE)) as response:
                response.raise_for_status()
        except requests.exceptions.RequestException as req_e:
            logger.error(f"Error during API request: {req_e}")
        except Exception as e:
            logger.error(f"Error sending OCR result: {e}")

    def run_ocr(self, min_conf=0.9, save_box_info=False, save_conf_info=False):
        # ---------------- Profiler Initialization ----------------
        # process = psutil.Process(os.getpid())
        
        # Prime CPU calculation & get baseline memory
        # process.cpu_percent(interval=None)
        # start_cpu_time = process.cpu_times()
        # ram_start_mb = process.memory_info().rss / (1024 * 1024)
        # t_init = time.perf_counter()

        img = None
        crop_img = None
        output = None
        try:
            img = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to load image")

            h, w = img.shape[:2]
            # Comment for debugging
            # self.save_30_percent_image(crop_img)
            reader = self.get_cached_reader()

            if h <= 100:
                output = reader(img)
            else:
                crop_img = img[int(h * 0.1):int(h * 0.4), 0:w]
                output = reader(crop_img)

            # Sample GPU usage immediately after inference
            # gpu_usage = get_current_gpu_usage()

            # # ---------------- Profiling Metrics ----------------
            # elapsed_time = time.perf_counter() - t_init
            # ram_end_mb = process.memory_info().rss / (1024 * 1024)
            # ram_increase_mb = max(0.0, ram_end_mb - ram_start_mb)

            # # Calculate total multi-threaded CPU usage across all cores
            # end_cpu_time = process.cpu_times()
            # total_cpu_seconds = (end_cpu_time.user - start_cpu_time.user) + (end_cpu_time.system - start_cpu_time.system)
            # # cpu_usage_pct = (total_cpu_seconds / elapsed_time * 100) if elapsed_time > 0 else 0.0
            
            # num_cores = psutil.cpu_count(logical=True) or 1
            # cpu_usage_pct = ((total_cpu_seconds / elapsed_time * 100) / num_cores) if elapsed_time > 0 else 0.0

            # # Print / Log formatted metrics
            # metrics_summary = (
            #     f"\n--- Resource Usage ---"
            #     f"\nRuntime      : {elapsed_time:.2f} s"
            #     f"\nCPU Usage    : {cpu_usage_pct:.1f}%"
            #     f"\nRAM Usage    : {ram_end_mb:.1f} MB"
            #     f"\nRAM Increase : {ram_increase_mb:.1f} MB"
            #     f"\nGPU Usage    : {gpu_usage}"
            #     f"\n----------------------"
            #     f"\n"
            # )
            # # print(metrics_summary)
            # logger.info(metrics_summary)

            # ---------------- OCR Output Preparation ----------------
            if not output:
                logger.info("[OCRText] No text detected")
                json_output = {"data": [{"text": "No text detected"}]}
                self._send_ocr_result(json_output)
            else:
                json_output = {"data": []}
                for box, text, conf in zip(output.boxes, output.txts, output.scores):
                    if conf < min_conf:
                        continue
                    json_data = {"text": text}
                    if save_conf_info:
                        json_data["confidence"] = float(conf)
                    if save_box_info:
                        json_data["box"] = [[float(p[0]), float(p[1])] for p in box]
                    json_output['data'].append(json_data)

                self._send_ocr_result(json_output)

        finally:
            del img
            del crop_img
            del output
            gc.collect()


if __name__ == "__main__":
    server_url = ""
    screenshot_id = ""
    image_path = "test.png"
    ActiveWindowOCRText(server_url, screenshot_id, image_path, warmup=True).run_ocr()
    