def parse_arff_to_csv(arff_file_path, csv_file_path):
    with open(arff_file_path, 'r') as arff_file, open(csv_file_path, 'w') as csv_file:
        # 读取文件内容，找到数据开始的位置
        data_start = False
        attributes = []
        for line in arff_file:
            line = line.strip()
            if line.startswith('@attribute'):
                # 提取属性名称
                attr_name = line.split()[1].strip("'")
                attributes.append(attr_name)
            elif line.lower() == '@data':
                # 数据开始
                data_start = True
                # 写入csv文件的表头
                csv_file.write(','.join(attributes) + '\n')
            elif data_start:
                # 写入数据行
                csv_file.write(line + '\n')

# 调用函数进行转换
parse_arff_to_csv('KDDTest+.arff', 'KDDTest+.csv')