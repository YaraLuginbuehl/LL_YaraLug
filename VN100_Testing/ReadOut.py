
def read_bin_file(file_path):
    import os 
    import pandas as pd
    
    parsed_data = []
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return None
    else:
        with open(file_path, "r") as f:
            for line in f:
                if line.startswith("$VNYMR"):
                    parts = line.strip().split(',')
                    if len(parts) == 13:  # 12 values + header
                        try:
                            yaw = float(parts[1])
                            pitch = float(parts[2])
                            roll = float(parts[3])
                            mag = list(map(float, parts[4:7]))
                            accel = list(map(float, parts[7:10]))
                            gyro = list(map(lambda x: float(x.split('*')[0]), parts[10:13]))

                            parsed_data.append({
                                "yaw": yaw,
                                "pitch": pitch,
                                "roll": roll,
                                "magX": mag[0],
                                "magY": mag[1],
                                "magZ": mag[2],
                                "accelX": accel[0],
                                "accelY": accel[1],
                                "accelZ": accel[2],
                                "gyroX": gyro[0],
                                "gyroY": gyro[1],
                                "gyroZ": gyro[2],
                            })
                        except ValueError:
                            continue
        df = pd.DataFrame(parsed_data)
        return df
