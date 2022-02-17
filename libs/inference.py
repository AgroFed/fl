def trim_data(data, trim_labels=[]):
    data = list(data)
    #print(len(data))

    keep_data = []
    for index, _ in enumerate(data):
        data[index] = list(data[index])
        if data[index][1] not in trim_labels:
            keep_data.append(tuple([data[index][0], data[index][1]]))

    #print(len(keep_data))
    return tuple(keep_data)