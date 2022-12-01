import re
import csv
import numpy as np
import os

# TODO: check, if any function of this file is needed.

def load_pts(pts_file):
    f = open(pts_file, 'r')

    data = {}
    name = os.path.splitext(os.path.basename(pts_file))[0]
    is_inside_values = False
    values = []
    float_regex_group = '([-+]?\d*\.\d+|\d+)'

    for line in f:
        if None != re.search('{', line):
            is_inside_values = True
        elif None != re.search('^}', line):
            is_inside_values = False
        elif is_inside_values:
            value_matches = re.finditer(float_regex_group + ' ' + float_regex_group, line)
            for value_match in value_matches:
                values.append(np.array((float(value_match.group(1)), float(value_match.group(2)))))

    data[name] = values

    return data


def load_pts_folder(in_folder):
    in_ext = '.pts'
    data = {}

    for f in os.listdir(in_folder):
        filename = os.path.join(in_folder, f)
        if os.path.isfile(filename) and f.endswith(in_ext):
            data_entry = load_pts(filename)
            data.update(data_entry)

    return data


def load_old_result(file, dim=2):
    f = open(file, 'r')

    data = {}
    groundtruth = {}
    float_regex_group = '([-+]?(?:\d*\.\d+|\d+)(?:[eE][+-]\d+)?)'
    #float_regex_group = '([-+]?\d*\.\d+|\d+|[-+]?\d*\.\d+[eE][+-]\d+)'
    #float_regex_group = '(([1-9][0-9]*\.?[0-9]*)|(\.[0-9]+))([Ee][+-]?[0-9]+)?'

    state = 0

    for line in f:
        if state == 0:
            name = line[:-1]
            state = 1
        elif state == 1:
            value_matches = re.finditer(float_regex_group + ',' + float_regex_group + ',' + float_regex_group, line)
            values = []
            count = 0
            for value_match in value_matches:
                count += 1
                if dim == 2:
                    values.append(np.array((float(value_match.group(1)), float(value_match.group(2)))))
                else:
                    values.append(np.array((float(value_match.group(1)), float(value_match.group(2)), float(value_match.group(3)))))
            if name in data:
                count = 0
                while (name + str(count)) in data:
                    count += 1
                data[name + str(count)] = values
            else:
                data[name] = values
            #print(name, count)
            state = 2
        elif state == 2:
            value_matches = re.finditer(float_regex_group + ',' + float_regex_group + ',' + float_regex_group, line)
            values = []
            count = 0
            for value_match in value_matches:
                count += 1
                if dim == 2:
                    values.append(np.array((float(value_match.group(1)), float(value_match.group(2)))))
                else:
                    values.append(np.array((float(value_match.group(1)), float(value_match.group(2)), float(value_match.group(3)))))
            if name in groundtruth:
                count = 0
                while (name + str(count)) in groundtruth:
                    count += 1
                groundtruth[name + str(count)] = values
            else:
                groundtruth[name] = values
            #print(name, count)
            state = 0

    return (data, groundtruth)


def load_multi_csv(filename, num_points, dim=2):
    points_dict = {}
    with open(filename, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            name = row[0] + '/' + row[1]
            #if int(row[1]) != 0:
            #    continue
            values = []
            #print(len(points_dict), name)
            for i in range(2, dim * num_points + 2, dim):
                #print(i)
                if dim == 2:
                    values.append(np.array((float(row[i]), float(row[i + 1]))))
                elif dim == 3:
                    values.append(np.array((float(row[i]), float(row[i + 1]), float(row[i + 2]))))
            points_dict[name] = values
    return points_dict


def load_multi_dict_csv(filename, num_points, dim=2):
    points_dict = {}
    with open(filename, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            name = row[0]

            if name in points_dict:
                points_dict_per_image = points_dict[name]
            else:
                points_dict_per_image = {}
                points_dict[name] = points_dict_per_image

            person_id = row[1]
            #if int(row[1]) != 0:
            #    continue
            values = []
            #print(len(points_dict), name)
            for i in range(2, dim * num_points + 2, dim):
                #print(i)
                if dim == 2:
                    values.append(np.array((float(row[i]), float(row[i + 1]))))
                elif dim == 3:
                    values.append(np.array((float(row[i]), float(row[i + 1]), float(row[i + 2]))))
            points_dict_per_image[person_id] = values
    return points_dict


def save_csv(point_dict, filename):
    with open(filename, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, values in sorted(point_dict.items()):
            #print(key, values)
            row = [key]
            for point in values:
                for i in range(point.size):
                    row.append(point[i])
            writer.writerow(row)


def save_dict_csv(dict, filename):
    with open(filename, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in sorted(dict.items()):
            writer.writerow([key, value])


def save_multiple_csv(point_dict, filename):
    with open(filename, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for name, persons in sorted(point_dict.items()):
            #print(key, values)
            for person, values in sorted(persons.items()):
                row = [name]
                row.append(person)
                for point in values:
                    for i in range(point.size):
                        row.append(point[i])
                writer.writerow(row)


def save_id_list(data, id_file):
    f = open(id_file, 'w')

    for key in sorted(data):
        string = key + '\n'
        f.write(string)


def save_id_weights_list(data, weights, id_file):
    f = open(id_file, 'w')

    for key in sorted(data):
        string = key + ' ' + str(weights[key]) + '\n'
        f.write(string)


def point_is_valid(point):
    for i in range(len(point)):
        if point[i] <= 0:
            return False
    return True


def rearrange_indizes(point_dict, indizes):
    new_point_dict = {}
    for key, value in point_dict.items():
        new_points = []
        for index in indizes:
            new_points.append(value[index])
        new_point_dict[key] = new_points
    return new_point_dict


def filter_valid_points(point_dict):
    new_point_dict = {}
    for key, value in point_dict.items():
        valid = True
        for point in value:
            if point_is_valid(point) == False:
                valid = False
                break
        if valid:
            new_point_dict[key] = value
    return new_point_dict


def save_points_idl(point_dict, num_heatmaps, filename, id_prefix='', id_postfix=''):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    with open(filename, 'w') as out_file:
        for key, values in sorted(point_dict.items()):
            id = id_prefix + key + id_postfix
            row = '"' + id + '": '
            for i in range(num_heatmaps):
                if i not in values:
                    row += '(-1,-1,0),'
                else:
                    coord = values[i][1]
                    row += '(' + str(coord[0]) + ',' + str(coord[1]) + ',' + '0),'
            row = row[:-1] + '\n'
            out_file.write(row)
