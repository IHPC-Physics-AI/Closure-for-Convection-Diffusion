import numpy as np
import os

def read_data(filename):
    start_data = False
    finish_data = False

    counter = 0
    with open(file=filename, mode='r') as file:
        data = {}
        for line in file:
            content = line.strip()
            if len(content)==0:
                continue
            
            if content[0]=='-' and not start_data:
                start_data = True
            elif content[0]=='-' and start_data:
                finish_data = True
            else:
                pass

            if start_data and not finish_data:
                position, value = content.split()
                data[position] = value
                counter += 1
    print(f"{counter} rows of data read")
    return data

def extract_xyz_to_array(data, x_range:tuple, y_range:tuple, z_range:tuple, yxz=False):
    grid = np.zeros((x_range[1]-x_range[0]+1, y_range[1]-y_range[0]+1, z_range[1]-z_range[0]+1))
    counter = 0
    for position, value in data.items():
        if 'x' in position and 'y' in position and 'z' in position:
            # assumes position is of the form x?y?z?
            coordinates = position.replace('x',',').replace('y',',').replace('z',',').split(',')
            x = int(coordinates[1])
            y = int(coordinates[2])
            z = int(coordinates[3])

            grid[x-x_range[0]][y-y_range[0]][z-z_range[0]] = round(float(value), ndigits=10)
            counter += 1
    if yxz:
        grid = grid.transpose((1,0,2))
    print(f"{counter} data points used for grid")
    print(f"Checksum: {np.sum(grid)}")
    return grid


# directory = "fluent_data_related/surf-avg-grid"
# for file in os.listdir(directory):
#     if file[-4:] != '.txt':
#         continue
#     data = read_data(filename=f"{directory}/{file}")
#     grid = extract_xyz_to_array(data, x_range=(1,50), y_range=(1,30), z_range=(1,1), yxz=True)
#     np.save(f"{directory}/{file[:-4]}", grid)


# for file in os.listdir(directory):
#     if ".txt" not in file:
#         continue

#     data = read_data(filename=f"{directory}/{file}")
#     with open(f"{directory}/values_only_{file[:-4]}.csv", "w") as file:
#         for position, value in data.items():
#             file.write(value)
#             file.write("\n")