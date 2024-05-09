class Electrometer:
    # electrometer = Electrometer(r'C:\Users\burtonlabuser\Desktop\example_datasheet.csv')
    '''
    Call this to create an Electrometer object, the name of which can be anything.
    If you need to change from the default COM port or baud rate for serial communication, do so with arguments when creating this object.
    The annotation attribute is used later in the render() method.
    '''
    
    def __init__(self, filepath: str, rate: int, unit: str):
        self.filepath = filepath
        self.ports = {'Electrometer' : ['COM1', 9600], 'Scale' : ['COM5', 2400], 
                      'SyringePump' : ['COM7', 19200], 'Arduino' : ['COM4', 9600]}
        self.rate = rate
        self.unit = unit
        self.annotation = None

        self.liveInterface()

    '''
    This liveInterface() method is called on the Electrometer object you've created to have it communicate with and live plot data from the Keithley 6514.
    Note that the recorded data will be saved to the absolute file path you provide, prefaced with r to take the string contents as unescaped characters.
    '''

    def liveInterface(self, delay_ms = 1000, plotLabelFontSize = 10, plotStyle = '.-r', annotColor = 'red'):
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import serial
        import time
        import pyArduino
        from IPython.display import display

        with (
            serial.Serial(self.ports['Electrometer'][0], self.ports['Electrometer'][1]) as serialElectrometer,
            serial.Serial(self.ports['Scale'][0], self.ports['Scale'][1]) as serialBalance,
            serial.Serial(self.ports['SyringePump'][0], self.ports['SyringePump'][1]) as serialSyringePump,
            pyArduino.Arduino(self.ports['Arduino'][0], self.ports['Arduino'][1]) as arduino
        ):
        
        # The script creates a serial object, ser, with our earlier arguments to be used going forward
            if (serialElectrometer.isOpen() == False): serialElectrometer.open()
            if (serialBalance.isOpen() == False): serialBalance.open()
            if (serialSyringePump.isOpen() == False): serialSyringePump.open()
            # Check that the COM port of our ser object is currently closed, and if so, open it for communication

            serialElectrometer.write("*RST; :SENS:FUNC 'CHAR'; CHAR:RANG:AUTO ON; :SYST:ZCH OFF; :FORM:ELEM READ\n".encode('utf-8'))
            # Commands to restore defaults, configure charge measurement with auto-range, disable zero check, and format to only return readings

            serialBalance.write("0A\r\n".encode('utf-8')); serialBalance.write("0M\r\n".encode('utf-8'))
            serialBalance.write("0S\r\n".encode('utf-8')); serialBalance.write("T\r\n".encode('utf-8'))
            # Commands to disable auto-print functionality, set unit to grams, and disable stability before zeroing the scale
            
            startTime = time.time()
            # Takes current UNIX time as t0
            timeSeries, chargeReadings, massReadings, humidityReadings, temperatureReadings = np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)
            # Declare single-valued "zero" arrays for use as our vectors

            try:
                fig = plt.figure(figsize = (16, 9), facecolor = 'xkcd:light gray')
                # Create our matplotlib figure object in 16:9 scale with a light gray background

                serialSyringePump.write(("FUN [RAT [" + str(self.rate) + " [" + self.unit + "]]]\r").encode('utf-8'))
                # Command to start pumping at the rate described in the arguments of the class definition
                
                while True:
                # Everything indented below is within our continuously-executed while loop:
                    timeSeries = np.append(timeSeries, (time.time() - startTime)) 
                    # Add new time step as UNIX time elapsed since t0

                    serialElectrometer.write("READ?\r".encode())
                    chargeReadings = np.append(chargeReadings, float(serialElectrometer.readline())) 
                    # Query the electrometer for a reading and add new reading--the float-converted ASCII response to our READ? query--to our data vector

                    serialBalance.write("P\r\n".encode())
                    massReadings = np.append(massReadings, float(serialBalance.readline()[:10])) 
                    # Query the scale for a reading and add new reading--the float-converted ASCII response to our P (Print) query--to our data vector

                    humidity = arduino.i2crequest(0x38, 0x00, 2)
                    humidityReadings = np.append(humidityReadings, (float(humidity[0] << 8 | humidity[1]) * 100 / 65535))

                    temperature = arduino.i2crequest(0x38, 0x02, 2)
                    temperatureReadings = np.append(temperatureReadings, (float(temperature[0] << 8 | temperature[1]) * 200 / 65535 - 50))
                    
                    self.render(timeSeries, chargeReadings, plotLabelFontSize, plotStyle, annotColor)
                    # Calls the render() method to plot our time and data vectors

                    display(fig)
                    # Display our figure object with the IPython cell renderer

                    df = pd.DataFrame({'time (s)': [timeSeries[-1]], 'charge (pC)': [chargeReadings[-1]], 'mass (g)': [massReadings[-1]], 'RH (%)': [humidityReadings[-1]], 'temperature (C)': [temperatureReadings[-1]]})
                    # Make a new dataframe out of the most recent timestamped data point with shape [time, data]
                    df.to_csv(self.filepath, mode = 'a', header = False, index = False)
                    # Write or append this dataframe row to the .csv
                    
                    time.sleep((delay_ms / 1000))
                    # Wait n milliseconds before taking the next data point

            except KeyboardInterrupt:
            # Upon pressing the Interrupt button in an interactive window, the script stops and prints a brief summary of the full .csv
                serialElectrometer.close(); serialBalance.close(); serialSyringePump.close(); arduino.close()
                print("Exiting...")
                print((pd.read_csv(self.filepath).describe(percentiles = []).map("{0:.2f}".format)))

    '''
    No need to manually call this render() function, as it's called in our while loop above to plot on and update our figure.
    '''
    
    def render(self, x, y, fontSize: int, plotStyle: str, color: str):
        import matplotlib.pyplot as plt
        from IPython.display import clear_output
        
        plt.plot(x, y*1e12, plotStyle)
        # Plot our data vector (scaled for pC) over time on our figure object
        plt.title('Cup Charge'); plt.xlabel('time (s)', fontsize = fontSize); plt.ylabel('charge (pC)', fontsize = fontSize)
        # Set figure title and axes labels
        
        y_value = "{:.2e}".format(y[-1] * 1e12)
        # Scale and format our latest data point to 2 decimal place scientific notation for live plot annotation
        
        if self.annotation is not None:
            self.annotation.set_text(f'Y: {y_value}')
            # Update the live plot annotation text label
        else:
            self.annotation = plt.text(0.98, 0.98, f'Y: {y_value}', 
                                       ha='right', va='top', fontsize = fontSize, transform = plt.gca().transAxes, 
                                       bbox = dict(facecolor = color, alpha = 0.5, edgecolor = color))
            # Create the live plot annotation text label in the top right corner of the figure with desired background formatting
            
        clear_output(wait = True)
        # Clear previous output from the IPython cell renderer so we only have one figure rendered
        plt.draw()
        # Update our figure

class Chamber:
    # chamber = Monitor(True, 'Dev2/ai0', '3A', r'C:\Users\burtonlabuser\Desktop\example_datasheet.csv')
    def __init__(self, correctionStatus: bool, target: str, editor: str, filepath: str, 
                 humidityPorts = ['Dev2/ai0', 'Dev2/ai1'], tempChannels = ['3A', '3B', '3C'], port = "COM1", baudrate = 9600):
        import numpy as np
        
        self.correctionStatus = correctionStatus
        self.target = target
        self.editor = editor
        self.filepath = filepath
        self.humidityPorts = humidityPorts
        self.tempChannels = tempChannels
        self.port = port
        self.baudrate = baudrate

        self.data = np.zeros((1, 1 + len(humidityPorts) + len(tempChannels)))
        self.rows, self.cols = 2, max(len(self.humidityPorts), len(self.tempChannels))

        self.livePlotter()
    
    def livePlotter(self, samples = 1000, sample_Rate = 1000, delay_ms = 0, plotLabelFontSize = 9):
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from matplotlib import gridspec
        import serial
        import nidaqmx
        from nidaqmx.constants import TerminalConfiguration, AcquisitionType, Edge
        import time
        from IPython.display import display

        with nidaqmx.Task() as task, serial.Serial(self.port, self.baudrate) as ser:
            if (ser.isOpen() == False): ser.open()
            for humidity_port in self.humidityPorts:
                task.ai_channels.add_ai_voltage_chan(humidity_port, max_val = 5, min_val = 0, terminal_config = TerminalConfiguration.RSE)
                task.timing.cfg_samp_clk_timing(sample_Rate, active_edge= Edge.RISING, sample_mode= AcquisitionType.FINITE, samps_per_chan = samples)
            
            t0 = time.time()
            text_annotations = ([None] * (self.rows + self.cols))

            try:
                fig = plt.figure(figsize = (15, 10), facecolor = 'xkcd:light gray')
                gs = gridspec.GridSpec(self.rows, self.cols, width_ratios = [1] * self.cols)

                while True:
                    row = np.zeros(self.data.shape[1])
                    row[0] = time.time() - t0

                    self.serial_stack(ser, self.tempChannels, row)
                    
                    row[1:len(self.humidityPorts) + 1] = np.abs(((np.mean(np.reshape(task.read(samples), [2, samples]), axis = 1) / 5) - 0.16) / 0.0062)
                    self.humidity_correction(row)

                    self.data = np.append(self.data, row.reshape(1, -1), axis=0)            
                    
                    self.subplotter(self.tempChannels, gs, plotLabelFontSize, text_annotations)
                    self.subplotter(self.humidityPorts, gs, plotLabelFontSize, text_annotations)

                    display(fig)

                    df = pd.DataFrame(row.reshape(1, -1), columns = (['Time (s)'] + self.humidityPorts + self.tempChannels))
                    df.to_csv(self.filepath, mode = 'a', header = False, index = False)
                    
                    time.sleep((delay_ms / 1000))

            except KeyboardInterrupt:
                print("Exiting...")
                print((pd.read_csv(self.filepath).describe().applymap("{0:.2f}".format)))

    def humidity_correction(self, array):
        if self.correctionStatus:
            target_idx = self.humidityPorts.index(self.target)
            editor_idx = self.tempChannels.index(self.editor)

            array[1 + target_idx] /= (1.0546 - (0.00216 * array[1 + len(self.humidityPorts) + editor_idx]))
    
    def serial_stack(self, serial_Object, channels, array):
        for idx, channel in enumerate(channels):
            query = channel + '?' + '\n'
            serial_Object.write(query.encode('utf-8'))
            array[1 + len(self.humidityPorts) + idx] = serial_Object.readline()

    def subplotter(self, list, gridspace, fontSize, annotations):
        import matplotlib.pyplot as plt
        from IPython.display import clear_output 

        for j, item in enumerate(list):
            if list == self.tempChannels:
                yLabel, header, style, yMax, yMin, r = 'Temperature ($^{\circ}C$)', 'Channel: ', '.-r', 55, 20, 0
                index = 1 + len(self.humidityPorts) + j
                y_value = round(self.data[-1, 1 + len(self.humidityPorts) + j], 2)
            if list == self.humidityPorts:
                yLabel, header, style, yMax, yMin, r = '%RH', 'Humidity Sensor: ', '.-b', 110, 0, 1
                index = 1 + j
                y_value = round(self.data[-1, 1 + j], 2)

            ax = plt.subplot(gridspace[r, j])
            ax.plot(self.data[1:, 0], self.data[1:, index], style)
            ax.set_xlabel('Time (s)', fontsize = fontSize)
            ax.set_ylabel(yLabel, fontsize = fontSize)
            ax.set_ylim(yMin, yMax)
            ax.set_title(header + item) 

            if annotations[(3*r + j)] is not None:
                annotations[(3*r + j)].set_text(f'Y: {y_value}')
            else:
                annotations[(3*r + j)] = ax.text(0.95, 0.95, f'Y: {y_value}', ha='right', va='top', transform = ax.transAxes)

        clear_output(wait = True)
        plt.draw()