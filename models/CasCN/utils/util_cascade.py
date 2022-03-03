import time

def get_time_weibo(msg_time, return_hour=True, return_minute=False):
    return

def read_cascades_weibo(config, observation_time=None, pre_times=None, filename=None):
    if observation_time is None:
        observation_time = config.observation_time
    if pre_times is None:
        pre_times = config.pre_times
    if filename is None:
        filename = config.cascades

    fliter_low = config.least_num  # fliter observation num less than this value
    fliter_up = 1000     # fliter observation num larger than this value
    fliter_label = 1000   # fliter label larger than this value

    cascades_total = {}
    with open(filename) as f:
        for line in f:
            parts = line.split("\t")
            if len(parts) != 5:
                print('wrong format!')
                continue
            cascadeID = parts[0]
            n_nodes = int(parts[3])
            path = parts[4].split(" ")
            if n_nodes != len(path) :
                print('wrong number of nodes', n_nodes, len(path))

            msg_pub_time = parts[2]

            observation_path = []
            labels = []
            edges = set()
            for i in range(len(pre_times)):
                labels.append(0)
            for p in path:
                nodes = p.split(":")[0].split("/")
                time_now = int(p.split(":")[1])
                if time_now < observation_time:
                    observation_path.append(",".join(nodes) + ":" + str(time_now))
                    for i in range(1, len(nodes)):
                        edges.add(nodes[i - 1] + ":" + nodes[i] + ":1")
                for i in range(len(pre_times)):
                    if time_now < pre_times[i]:
                        labels[i] += 1

            if labels[0] > fliter_label:  # most are noisy or wrong dataset
                continue


            try:
                ts = time.strptime(parts[2], '%Y-%m-%d-%H:%M:%S')
                hour = ts.tm_hour
            except:
                msg_time = time.localtime(int(parts[2]))
                hour = time.strftime("%H",msg_time)
                hour = int(hour)

            if len(observation_path) < config.least_num and len(observation_path)>100:
                continue

            if hour <6 or hour > 21:
                continue

            cascades_total[cascadeID] = hour

