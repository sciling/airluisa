import GPUtil
from threading import Thread
import time
import utils_detec


class Monitor(Thread):
    def __init__(self, delay, output_path):
        super(Monitor, self).__init__()
        self.stopped = False
        self.output_path = output_path
        self.delay = delay  # Time between calls to GPUtil
        self.start()

    def run(self):
        # deviceIds = []
        gpuUtil = []
        memUtil = []
        out = self.output_path.split(".")[0]
        while not self.stopped:
            GPUs = GPUtil.getGPUs()
            for gpu in GPUs:
                # deviceIds.append(gpu.id)
                gpuUtil.append(gpu.load * 100)
                memUtil.append(gpu.memoryUtil * 100)

            # GPUtil.showUtilization()
            time.sleep(self.delay)

        avg_gpu_util, avg_gpu_mem = utils_detec.build_monitor_gpu_results(
            gpuUtil, memUtil
        )

        with open(out + "GPU_resumen.txt", "w") as f:
            f.write("Average use of GPU util " + str(avg_gpu_util) + " %\n")
            f.write("Average use of GPU mem util: " + str(avg_gpu_mem) + " %\n")

        print("Average use of GPU util: ", avg_gpu_util, " %")
        print("Average use of GPU mem util: ", avg_gpu_mem, " %")

    def stop(self):
        self.stopped = True
