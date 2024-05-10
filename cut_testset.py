# def process_text_file(input_file, output_file):
#     with open(input_file, 'r') as f:
#         lines = f.readlines()

#     with open(output_file, 'w') as f:
#         for i, line in enumerate(lines):
#             if i % 20 == 0:
#                 f.write(line)

# if __name__ == "__main__":
#     input_file = 'data/ucf101/ucfTrainTestlist/testlist01.txt'  # 输入文件名
#     output_file = 'data/ucf101/ucfTrainTestlist/testlist04.txt'  # 输出文件名
#     process_text_file(input_file, output_file)

import numpy
motion = numpy.load('00000001_00000003.npy')
print(motion.shape)