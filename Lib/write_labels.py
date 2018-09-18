def write_labels(label_map_name, pairs):
    with open(label_map_name,'w') as f:
        for x in pairs:
            f.write('item {\n')
            f.write('\tid: {}\n'.format(x[0]))
            f.write('\tname: \'{}\'\n'.format(x[1]))
            f.write('}\n')