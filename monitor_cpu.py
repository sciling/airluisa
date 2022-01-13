import psutil
from threading import Thread
import time
import utils_detec


class Monitor_CPU(Thread):
    def __init__(self, delay, output_path):
        super(Monitor_CPU, self).__init__()
        self.stopped = False
        self.output_path = output_path
        self.delay = delay # Time between calls to psutil
        self.start()

    def run(self):
        cpu_usage = []
        cpu_freq = []
        ram_usage = []
        out = self.output_path.split(".")[0]
        while not self.stopped:

            cpu_usage.append(psutil.cpu_percent(interval=0.5))
            cpu_freq.append(int(psutil.cpu_freq().current))
            ram_usage.append(int(int(psutil.virtual_memory().total - psutil.virtual_memory().available)) / 1024 / 1024)


            time.sleep(self.delay)
        
        avg_cpu_usage, avg_cpu_freq, avg_ram_usage = utils_detec.build_monitor_cpu_results(cpu_usage, cpu_freq, ram_usage)
        
        with open(out+"CPU_resumen.txt", "w") as f:
            f.write("Average use of CPU: "+ str(avg_cpu_usage) + " %\n")
            f.write("Average frequency use of CPU: "+ str(avg_cpu_freq) + " MHz\n")
            f.write("Average memory RAM: "+ str(avg_ram_usage) +  " MB\n")
            f.write("RAM total is "+ str(int(int(psutil.virtual_memory().total) / 1024 / 1024)))

        print("Average use of CPU: ", avg_cpu_usage, " %")
        print("Average frequency use of CPU: ", avg_cpu_freq, " MHz")
        print("Average memory RAM: ", avg_ram_usage, " MB")

    def stop(self):
        self.stopped = True

