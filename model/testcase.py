# -*- coding: utf-8 -*-
"""
This module defines the API to the test case used by the REST requests to 
perform functions such as advancing the simulation, retreiving test case 
information, and calculating and reporting results.

"""

from fmpy import *
import cus_simulation
import numpy as np
import copy
import json
import time



def _process_input(u, start_time):
    '''Convert the input dictionary into a structured array.
        
    Parameters
    ----------
    u : dict
        Defines the control input data to be used for the step.
        {<input_name> : <input_value>}
        
    start_time: int
        Start time of simulation in seconds.
            
        
    Returns
    -------
    input_object : structured array
        Input for next time step
            
    '''    
        
    if u.keys():
        # If there are overwriting keys available
        # Check that any are overwritten
        written = False
        for key in u.keys():
            if u[key]:
                written = True
                break
        # If there are, create input object
        if written:
            dtype = [('time', np.double)]   
            values = [start_time]
            for key in u.keys():    
                if isinstance(u[key], (int, float)):       
                    dtype.append((key, np.double))        
                else:          
                    dtype.append((key, np.bool_))
                values.append(u[key])    

            input_object = np.array([tuple(values)], dtype=dtype)
        # Otherwise, input object is None
        else:
            input_object = None    
        # Otherwise, input object is None
    else:
        input_object = None  

    return input_object   


class TestCase(object):
    '''Class that implements the test case.
    
    '''
    
    def __init__(self,con):
        '''Constructor.
        
        '''
        
        # Define simulation model
        self.fmupath = con['fmupath']
        tempdir = extract(self.fmupath)
        # Load fmu                
        model_description = read_model_description(self.fmupath)
        self.model_description = model_description
        self.fmu = instantiate_fmu(tempdir, model_description)
        self.fmu.setupExperiment(startTime=0)
        self.fmu.enterInitializationMode()
        self.fmu.exitInitializationMode()
#        self.fmu.enterContinuousTimeMode()         

        # Get version and check is 2.0
        self.fmu_version = model_description.fmiVersion
        if self.fmu_version != '2.0':
            raise ValueError('FMU must be version 2.0.')

        # Get available control inputs and outputs
        self.input_names = []
        self.output_names = []

        for v in model_description.modelVariables:
            if v.causality == 'input':
                self.input_names.append(v.name)
            if v.causality == 'output':
                self.output_names.append(v.name)  

        # Set default communication step
        self.set_step(con['step'])


        # Set initial fmu simulation start
        self.start_time = 0

        # Initialize simulation data arrays
        self.__initilize_data()

    def __initilize_data(self):
        '''Initializes objects for simulation data storage.
        
        Uses self.output_names and self.input_names to create
        self.y, self.y_store, self.u, and self.u_store.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        '''
    
        # Outputs data
        self.y = {'time':[]}
        for key in self.output_names:
            self.y[key] = []
        self.y_store = copy.deepcopy(self.y)
        # Inputs data
        self.u = {'time':[]}
        for key in self.input_names:
            self.u[key] = []        
        self.u_store = copy.deepcopy(self.u)
                
    def __simulation(self,start_time,end_time,input_object=None):
        '''Simulates the FMU using the pyfmi fmu.simulate function.
        
        Parameters
        ----------
        start_time: int
            Start time of simulation in seconds.
        final_time: int
            Final time of simulation in seconds.
        input_object: pyfmi input_object, optional
            Input object for simulation
            Default is None
        
        Returns
        -------
        res: pyfmi results object
            Results of the fmu simulation.
        
        '''

        res = cus_simulation.simulate_fmu(filename = self.fmupath,
                                     start_time=start_time, 
                                     stop_time=end_time,
                                     output= self.input_names + self.output_names,
                                     input = input_object,                                     
                                     fmu_instance=self.fmu)                                    

        return res            

    def __get_results(self, res, store=True):
        '''Get results at the end of a simulation and throughout the 
        simulation period for storage. This method assigns these results
        to `self.y` and, if `store=True`, also to `self.y_store` and 
        to `self.u_store`. 
        This method is used by `initialize()` and `advance()` to retrieve
        results. `initialize()` does not store results whereas `advance()`
        does. 
        
        Parameters
        ----------
        res: pyfmi results object
            Results of the fmu simulation.
        store: boolean
            Set to true if desired to store results in `self.y_store` and
            `self.u_store`
        
        '''
        
        # Get result and store measurement
        for key in self.y.keys():
            self.y[key] = res[key][-1]
            if store:
                self.y_store[key] = self.y_store[key] + res[key].tolist()[1:]
        
        # Store control inputs
        if store:
            for key in self.u.keys():
                self.u_store[key] = self.u_store[key] + res[key].tolist()[1:] 

    def advance(self,u):
        '''Advances the test case model simulation forward one step.
        
        Parameters
        ----------
        u : dict
            Defines the control input data to be used for the step.
            {<input_name> : <input_value>}
            
        Returns
        -------
        y : dict
            Contains the measurement data at the end of the step.
            {<measurement_name> : <measurement_value>}
            
        '''
            
        # Set final time
        self.final_time = self.start_time + self.step
        # Set control inputs if they exist and are written
        # Check if possible to overwrite
        input_object = _process_input(u, self.start_time)
        # Simulate
#        print(input_object)
        res = self.__simulation(self.start_time,self.final_time,input_object) 

        # Process results
        if res is not None:        
            # Get result and store measurement and control inputs
            self.__get_results(res, store=True)
            # Advance start time
            self.start_time = self.final_time
            # Raise the flag to compute time lapse
            self.tic_time = time.time()

            return self.y

        else:

            return None        

    def initialize(self, start_time, warmup_period):
        '''Initialize the test simulation.
        
        Parameters
        ----------
        start_time: int
            Start time of simulation to initialize to in seconds.
        warmup_period: int
            Length of time before start_time to simulate for warmup in seconds.
            
        Returns
        -------
        y : dict
            Contains the measurement data at the end of the initialization.
            {<measurement_name> : <measurement_value>}

        '''

        # Reset fmu
        self.fmu.reset()              
        self.fmu.setupExperiment(startTime=max(start_time-warmup_period,0))
        self.fmu.enterInitializationMode()
        self.fmu.exitInitializationMode()
        self.fmu.enterContinuousTimeMode()                        
        # Reset simulation data storage
        self.__initilize_data()            
        res = self.__simulation(max(start_time-warmup_period,0), start_time)
        # Process result
        if res is not None:
            # Get result
            self.__get_results(res, store=False)
            # Set internal start time to start_time
            self.start_time = start_time

            return self.y
        
        else:

            return None
        
    def get_step(self):
        '''Returns the current simulation step in seconds.'''

        return self.step

    def set_step(self,step):
        '''Sets the simulation step in seconds.
        
        Parameters
        ----------
        step : int
            Simulation step in seconds.
            
        Returns
        -------
        None
        
        '''
        
        self.step = float(step)
        
        return None
        
    def get_inputs(self):
        '''Returns a dictionary of control inputs and their meta-data.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        inputs : dict
            Dictionary of control inputs and their meta-data.
            
        '''

        inputs = self.input_names
        
        return inputs
        
    def get_measurements(self):
        '''Returns a dictionary of measurements and their meta-data.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        measurements : dict
            Dictionary of measurements and their meta-data.
            
        '''

        measurements = self.output_names
        
        return measurements
        
    def get_results(self):
        '''Returns measurement and control input trajectories.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        Y : dict
            Dictionary of measurement and control input names and their 
            trajectories as lists.
            {'y':{<measurement_name>:<measurement_trajectory>},
             'u':{<input_name>:<input_trajectory>}
            }
        
        '''
        
        Y = {'y':self.y_store, 'u':self.u_store}
        
        return Y
        


        
    def get_name(self):
        '''Returns the name of the test case fmu.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        name : str
            Name of test case fmu.
            
        '''
        
        name = self.fmupath[7:-4]
        
        return name
        
         