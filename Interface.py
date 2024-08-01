class Electrometer:
    # elec = Electrometer(r'C:\Users\burtonlabuser\Desktop\drops-data\example_datasheet.csv', 'g', {'p1' : [500, 'um']})
    '''
    Call this to create and record data with an Electrometer object, the name of which can be anything.
    If you need to change from the default COM port or baud rate for serial communication, change the ports dict in __init__ when creating this object.
    '''
    def __init__(self, filepath: str, syringe_type : str, program : dict[str : [float, str]]):
        self.filepath = filepath; self.annotation = None
        self.ports = {'Electrometer' : ['COM1', 9600], 'NewScale' : ['COM17', 19200], 'SyringePump' : ['COM15', 19200], 'Arduino' : ['COM4', 9600]}
        self.syrdiam = 21.46 if syringe_type == 'g' else 20.1
        
        match len(program):
            case 1:
                self.phases = 1
                self.rate = program['p1'][0]
                self.unit = (program['p1'][1]).lower()
                match self.unit:
                    case 'um': self.formUnit = f'${self.rate}\,\mu l/min$'; self.convertedRate = self.rate * (1/(1000*60))
                    case 'mm': self.formUnit = f'${self.rate}\,ml/min$'; self.convertedRate = self.rate * (1/60)
                    case 'uh': self.formUnit = f'${self.rate}\,\mu l/hr$'; self.convertedRate = self.rate * (1/(1000*3600))
                    case 'mh': self.formUnit = f'${self.rate}\,\ml/hr$'; self.convertedRate = self.rate * (1/(3600)) 
                    case _: print("Incorrect pump rate unit argument. Options are 'UM' ($\mu l/min$), 'MM' ($ml/min$), 'UH' ($\mu l/hr$), and 'MH' ($ml/hr$).")
                self.pumpCommands = ["dia" + str(self.syrdiam), "phn1", "funbep", "funrat", "rat" + str(self.rate) + self.unit, "vol2.0", "dirinf", "run1"]
            case 2:
                self.phases = 2
                self.rate = [program['p1'][0], program['p2'][0]]
                self.unit = [(program['p1'][1]).lower(), (program['p2'][1]).lower()]; self.formUnit = ['', '']; self.convertedRate = [0, 0]
                for i in range(2):
                    match self.unit[i]:
                        case 'um': self.formUnit[i] = f'${self.rate[i]}\,\mu l/min$'; self.convertedRate[i] = self.rate[i] * (1/(1000*60))
                        case 'mm': self.formUnit[i] = f'${self.rate[i]}\,ml/min$'; self.convertedRate[i] = self.rate[i] * (1/60)
                        case 'uh': self.formUnit[i] = f'${self.rate[i]}\,\mu l/hr$'; self.convertedRate[i] = self.rate[i] * (1/(1000*3600))
                        case 'mh': self.formUnit[i] = f'${self.rate[i]}\,\ml/hr$'; self.convertedRate[i] = self.rate[i] * (1/(3600)) 
                        case _: print("Incorrect pump rate unit argument. Options are 'UM' ($\mu l/min$), 'MM' ($ml/min$), 'UH' ($\mu l/hr$), and 'MH' ($ml/hr$).")
                self.pumpCommands = ["dia" + str(self.syrdiam), "phn1", "funrat", "rat" + str(self.rate[0]) + self.unit[0], "vol2.0", "dirinf", "run1", "stp",
                                    "phn2", "funrat", "rat" + str(self.rate[1]) + self.unit[1], "vol2.0", "dirinf", "run2"]
        
        self.liveInterface()
    
    def liveInterface(self, loopdelay_s = 0.001, comdelay_s = 0.01, plotArgs = {'fontsize' : 10, 'style' : '.-r', 'annotcolor' : 'r'}):
        '''
        This liveInterface() method is called on the Electrometer object you've created to have it communicate with and live plot data from the Keithley 6514.
        Note that the recorded data will be saved to the absolute file path you provide, prefaced with r to take the string contents as unescaped characters.
        '''
        import pandas as pd
        import matplotlib.pyplot as plt
        import serial, time
        from IPython.display import display

        with (serial.Serial(self.ports['Electrometer'][0], self.ports['Electrometer'][1]) as serialElectrometer,
            serial.Serial(self.ports['NewScale'][0], self.ports['NewScale'][1]) as serialBalance,
            serial.Serial(self.ports['SyringePump'][0], self.ports['SyringePump'][1]) as serialSyringePump,
            serial.Serial(self.ports['Arduino'][0], self.ports['Arduino'][1]) as arduino):
            # The script creates a serial object for each external device with our earlier arguments
        
            try:
                #fig = plt.figure(figsize = (16, 9), facecolor = 'xkcd:light gray')
                # Create our matplotlib figure object in 16:9 scale with a light gray background

                timeSeries = [0]; chargeReadings = [0]; massReadings = [0]; humidityReadings = [0]; temperatureReadings = [0]
                # Declare empty lists for use as our time and data vectors

                serialElectrometer.write(b"*RST; :SYST:ZCH ON; :SENS:FUNC 'CHAR'; CHAR:RANG:UPP 20e-9; :FORM:ELEM READ; :SYST:ZCH OFF; :CALC2:NULL:STAT ON\n")
                #serialElectrometer.write(b"*RST; :SYST:ZCH ON; :SENS:FUNC 'CHAR'; CHAR:RANG:AUTO ON; :FORM:ELEM READ; :SYST:ZCH OFF; :CALC2:NULL:STAT ON\n")
                # Commands to restore defaults, enable zero check, configure charge measurement with the 20nC range, format to only return readings, disable zero check, and set relative

                #for command in ["0A", "0M", "0S", "T"]:
                #    serialBalance.write(command.encode() + b"\r\n")
                #    time.sleep(comdelay_s)
                # Commands to disable auto-print functionality, set unit to grams, and disable stability before zeroing the scale

                serialBalance.write(b"Z\r\n")
                
                t0 = time.time()
                # Define t0
                
                if self.phases == 1:
                    while (timeSeries[-1] < 30):
                        timeSeries.append(time.time() - t0)
                        # Add new time step as UNIX time elapsed since t0

                        self.comsideration(serialElectrometer, chargeReadings)
                        self.comsideration(serialBalance, massReadings)

                        humidityReadings.append(0); temperatureReadings.append(0)

                    for command in self.pumpCommands:
                        serialSyringePump.write(command.encode() + b"\r")
                        time.sleep(comdelay_s)
                    # Commands to start pump infusion program at the rate described in the arguments of the class definition

                    while (timeSeries[-1] < 450):
                        timeSeries.append(time.time() - t0)
                        # Add new time step as UNIX time elapsed since t0

                        self.comsideration(serialElectrometer, chargeReadings)
                        self.comsideration(serialBalance, massReadings)
                        self.comsideration(arduino, [temperatureReadings, humidityReadings])
                        
                        #self.render(data_vectors = [timeSeries, chargeReadings, massReadings, humidityReadings, temperatureReadings], fontSize = plotLabelFontSize, plotStyle = plotStyle, color = annotColor)
                        # Calls the render() method to plot our time and data vectors

                        #display(fig)
                        # Display our figure object with the IPython cell renderer
                        
                        time.sleep(loopdelay_s)
                        # Wait n seconds before taking the next data point
                else:
                    while (timeSeries[-1] < 30):
                        timeSeries.append(time.time() - t0)
                        # Add new time step as UNIX time elapsed since t0

                        self.comsideration(serialElectrometer, chargeReadings)
                        self.comsideration(serialBalance, massReadings)

                        humidityReadings.append(0); temperatureReadings.append(0)

                    for command in self.pumpCommands[:7]:
                        serialSyringePump.write(command.encode() + b"\r")
                        time.sleep(comdelay_s)
                    # Commands to start pump infusion program at the rate described in the arguments of the class definition

                    while (timeSeries[-1] < 450):
                        timeSeries.append(time.time() - t0)
                        # Add new time step as UNIX time elapsed since t0

                        self.comsideration(serialElectrometer, chargeReadings)
                        self.comsideration(serialBalance, massReadings)
                        self.comsideration(arduino, [temperatureReadings, humidityReadings])
                        
                        #self.render(data_vectors = [timeSeries, chargeReadings, massReadings, humidityReadings, temperatureReadings], fontSize = plotLabelFontSize, plotStyle = plotStyle, color = annotColor)
                        # Calls the render() method to plot our time and data vectors

                        #display(fig)
                        # Display our figure object with the IPython cell renderer
                        
                        time.sleep(loopdelay_s)
                        # Wait n seconds before taking the next data point

                    for command in self.pumpCommands[7:]:
                        serialSyringePump.write(command.encode() + b"\r")
                        time.sleep(comdelay_s)
                    # Commands to start pump infusion program at the rate described in the arguments of the class definition

                    while (timeSeries[-1] < 1000):
                        timeSeries.append(time.time() - t0)
                        # Add new time step as UNIX time elapsed since t0

                        self.comsideration(serialElectrometer, chargeReadings)
                        self.comsideration(serialBalance, massReadings)
                        self.comsideration(arduino, [temperatureReadings, humidityReadings])
                        
                        #self.render(data_vectors = [timeSeries, chargeReadings, massReadings, humidityReadings, temperatureReadings], fontSize = plotLabelFontSize, plotStyle = plotStyle, color = annotColor)
                        # Calls the render() method to plot our time and data vectors

                        #display(fig)
                        # Display our figure object with the IPython cell renderer
                        
                        time.sleep(loopdelay_s)
                        # Wait n seconds before taking the next data point
                    

            except KeyboardInterrupt: pass
            finally:
                serialElectrometer.close(); serialBalance.close(); arduino.close()
                serialSyringePump.write(b'stp\r'); serialSyringePump.close()

                df = pd.DataFrame(dict([(keys, pd.Series(values)) for keys, values 
                                  in {"time (s)": timeSeries, "charge (pC)": chargeReadings, "mass (g)": massReadings, "RH (%)": humidityReadings, "temp (C)": temperatureReadings}.items()]))
                    
                df.to_csv(self.filepath, mode = 'a', header = True, index = False)
                # Assemble a dataframe from the data vectors and save to disk as a .csv
                
                print("Exiting...", f"Data is located at {self.filepath}.")
                # Upon pressing the Interrupt button in an interactive window, or the duration of the program completes, the script stops and prints path for the .csv
    
    def comsideration(self, serial_device, data_vector : list | list[list]):
        '''
        This `comsideration()` function is called in the `liveInterface()` while loop to communicate with and capture readings from our serial instruments.
        '''
        import numpy as np
        import re

        if (serial_device.port == self.ports['Electrometer'][0]):
            serial_device.write(b"READ?\r")
            data_vector.append(float(serial_device.readline()) * 1e12)
            # Query the electrometer for a reading and add new reading--the float-converted ASCII response to our READ? query--to our data vector
        
        elif (serial_device.port == self.ports['NewScale'][0]):
            serial_device.write(b"IP\r\n")
            mass = serial_device.readline()[1:11]
            if (mass == b"ES\r\n" or mass == b"" or mass == b"OK!\r\n" or mass == b"K!\r\n"): mass = np.nan
            data_vector.append(float(mass))
            # Query the scale for a reading and add new reading--the float-converted ASCII response to our P (Print) query--to our data vector
        
        elif (serial_device.port == self.ports['Arduino'][0]):
            ahtReadout = (serial_device.read_until(b"\r\n")).split(b"\r\n")[0].decode()
            try: 
                temp, hum = ahtReadout[:ahtReadout.index(';')], ahtReadout[ahtReadout.index(';')+1:]
                if (len(temp) > 5 or temp == '' or (len(re.findall("\.", temp)) > 1)): temp = np.nan
                if (len(hum) > 5 or hum == '' or (len(re.findall("\.", hum)) > 1)): hum = np.nan
            except ValueError: temp, hum = np.nan, np.nan
            finally:
                data_vector[0].append(float(temp))
                data_vector[1].append(float(hum))
            # Read and parse the line printed from the Arduino Uno-connected AHT20 chip for humidity and temperature values
        
        else:
            print('Not a serial-connected device.')
    
    def render(self, data_vectors: list, style_arguments: dict):
        '''
        This `render()` function is called in our while loop to plot on and update our figure.
        '''
        import matplotlib.pyplot as plt
        from IPython.display import clear_output
        
        plt.plot(data_vectors[0][-1], (data_vectors[1][-1]), style_arguments['style'])
        # Plot our data vector (scaled for pC) over time on our figure object
        #plt.axvline(x = 60, color = 'g', alpha = 0.8, ls = '--')
        plt.title('Cup Charge'); plt.xlabel('time (s)', fontsize = style_arguments['fontsize']); plt.ylabel('charge (pC)', fontsize = style_arguments['fontsize'])
        # Set figure title and axes labels

        plt.subplots_adjust(top = 0.85)
        plt.figtext(0.128, 0.87, 
                    f"Temperature: {data_vectors[4][-1]:.2f}$\,^{{\circ}}\,\\text{{C}}$\nRelative Humidity: {data_vectors[3][-1]}%\nMass: {data_vectors[2][-1]:.2f}$\,\\text{{g}}$\nPump Rate: {self.formUnit}", 
                    fontsize = style_arguments['fontsize'], color = 'w', bbox = {'facecolor' : 'k', 'alpha' : 0.9, 'boxstyle' : 'round'})
        # Pad the top of the figure and add live-updating value readout for extras
        
        y_value = "{:.2e}".format(data_vectors[0][-1])
        # Scale and format our latest data point to 2 decimal place scientific notation for live plot annotation
        
        if self.annotation is not None:
            self.annotation.set_text(f'Y: {y_value}')
            # Update the live plot annotation text label
        else:
            self.annotation = plt.text(0.98, 0.98, f'Y: {y_value}', 
                                       ha='right', va='top', fontsize = style_arguments['fontsize'], transform = plt.gca().transAxes, 
                                       bbox = dict(facecolor = style_arguments['annotcolor'], alpha = 0.5, edgecolor = style_arguments['annotcolor']))
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