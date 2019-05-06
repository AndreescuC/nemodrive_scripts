import datetime


def main():
    files = ['means_24']#, 'means_session_0', 'means_session_1', 'means_session_2']
    path_to_means = '/home/andi/Sandbox/AIMAS/Scripts/landmark_visualizer/andia/rvec_logs/means_backup/'

    for file in files:
        write_means(path_to_means + file)


def write_means(path):
    f = open(path, 'a+')
    f.seek(0)
    means = {}
    current_camera = ''

    content = [x.strip() for x in f.readlines()]
    for line in content:
        if not line:
            continue
        if 'camera ' in line:
            current_camera = line.split('(')[0]
            means[current_camera] = {
                'rvec': [],
                'tvec': []
            }
            continue

        rvec_values, tvec_values = parse_values_line(line)
        means[current_camera]['rvec'].append(rvec_values)
        means[current_camera]['tvec'].append(tvec_values)

    means = compute_means(means)
    log_means(means, path)


def compute_means(means):
    result = {}
    for camera, data in means.items():
        rvec_means = list(map(
            lambda data1, data2, data3, data4: (data1 + data2 + data3 + data4) / 4,
            *data['rvec']
        ))
        tvec_means = list(map(
            lambda data1, data2, data3, data4: (data1 + data2 + data3 + data4) / 4,
            *data['tvec']
        ))
        result[camera] = {'rvec': rvec_means, 'tvec': tvec_means}

    return result


def log_means(means, path_to_means):
    path_to_means += "_result.txt"
    f = open(path_to_means, 'w')

    f.write("Means determined %s: " % datetime.date.today().strftime("%B %d, %Y"))
    for camera, data in means.items():
        rvec = data['rvec']
        tvec = data['tvec']
        f.write(
            "\nCamera %s:\n\trvec: [%f %f %f]\n\ttvec: [%f %f %f]\n" %
            (camera, rvec[0], rvec[1], rvec[2], tvec[0], tvec[1], tvec[2])
        )

    f.close()


def parse_values_line(line):
    values = line.split(';')

    rvec_values = [float(value) for value in values[0].replace('[', '').replace(']', '').split(' ')]

    tvec_values = values[1].split(' vs ')[0]
    tvec_values = [float(value) for value in tvec_values.replace('[', '').replace(']', '').split(' ')]

    return rvec_values, tvec_values


if __name__ == '__main__':
    main()
