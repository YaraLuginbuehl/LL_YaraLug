import os
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))


from ReadOut import read_bin_file
test1 = read_bin_file("test1.bin")

plt.figure(figsize=(10, 6))
plt.plot(test1['accelX'], label='Acceleration X', color='blue')
plt.plot(test1['accelY'], label='Acceleration Y', color='green')
plt.plot(test1['accelZ'], label='Acceleration Z', color='red')  
plt.title('Acceleration Data') 
plt.xlabel('Sample')
plt.ylabel('Acceleration (m/s²)')
plt.legend()
plt.grid()
plt.show()

