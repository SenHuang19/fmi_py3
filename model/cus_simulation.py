# noinspection PyPep8

import shutil
from fmpy.fmi1 import *
from fmpy.fmi1 import _FMU1
from fmpy.fmi2 import *
from fmpy.simulation import *
from fmpy.fmi2 import _FMU2
from fmpy import fmi3
from fmpy import extract
from fmpy.util import auto_interval, _is_string
import numpy as nps
from time import time as current_time

# absolute tolerance for equality when comparing two floats
eps = 1e-13




def simulate_fmu(filename,
                 validate=True,
                 start_time=None,
                 stop_time=None,
                 solver='CVode',
                 step_size=None,
                 relative_tolerance=None,
                 output_interval=None,
                 record_events=True,
                 fmi_type=None,
                 start_values={},
                 apply_default_start_values=False,
                 input=None,
                 output=None,
                 timeout=None,
                 debug_logging=False,
                 visible=False,
                 logger=None,
                 fmi_call_logger=None,
                 step_finished=None,
                 model_description=None,
                 fmu_instance=None):
    """ Simulate an FMU

    Parameters:
        filename            filename of the FMU or directory with extracted FMU
        validate            validate the FMU
        start_time          simulation start time (None: use default experiment or 0 if not defined)
        stop_time           simulation stop time (None: use default experiment or start_time + 1 if not defined)
        solver              solver to use for model exchange ('Euler' or 'CVode')
        step_size           step size for the 'Euler' solver
        relative_tolerance  relative tolerance for the 'CVode' solver and FMI 2.0 co-simulation FMUs
        output_interval     interval for sampling the output
        record_events       record outputs at events (model exchange only)
        fmi_type            FMI type for the simulation (None: determine from FMU)
        start_values        dictionary of variable name -> value pairs
        apply_default_start_values  apply the start values from the model description
        input               a structured numpy array that contains the input (see :class:`Input`)
        output              list of variables to record (None: record outputs)
        timeout             timeout for the simulation
        debug_logging       enable the FMU's debug logging
        visible             interactive mode (True) or batch mode (False)
        fmi_call_logger     callback function to log FMI calls
        logger              callback function passed to the FMU (experimental)
        step_finished       callback to interact with the simulation (experimental)
        model_description   the previously loaded model description (experimental)
        fmu_instance        the previously instantiated FMU (experimental)

    Returns:
        result              a structured numpy array that contains the result
    """

    from fmpy import supported_platforms
    from fmpy.model_description import read_model_description

    platforms = supported_platforms(filename)

    # use 32-bit DLL remoting
    use_remoting = platform == 'win64' and 'win64' not in platforms and 'win32' in platforms

    if fmu_instance is None and platform not in platforms and not use_remoting:
        raise Exception("The current platform (%s) is not supported by the FMU." % platform)

    if model_description is None:
        model_description = read_model_description(filename, validate=validate)
    else:
        model_description = model_description

    if fmi_type is None:
        if fmu_instance is not None:
            # determine FMI type from the FMU instance
            fmi_type = 'CoSimulation' if type(fmu_instance) in [FMU1Slave, FMU2Slave, fmi3.FMU3Slave] else 'ModelExchange'
        else:
            # determine the FMI type automatically
            fmi_type = 'CoSimulation' if model_description.coSimulation is not None else 'ModelExchange'

    if fmi_type not in ['ModelExchange', 'CoSimulation']:
        raise Exception('fmi_type must be one of "ModelExchange" or "CoSimulation"')

    experiment = model_description.defaultExperiment

    if start_time is None:
        if experiment is not None and experiment.startTime is not None:
            start_time = experiment.startTime
        else:
            start_time = 0.0

    if stop_time is None:
        if experiment is not None and experiment.stopTime is not None:
            stop_time = experiment.stopTime
        else:
            stop_time = start_time + 1.0

    if relative_tolerance is None and experiment is not None:
        relative_tolerance = experiment.tolerance

    if step_size is None:
        total_time = stop_time - start_time
        step_size = 10 ** (np.round(np.log10(total_time)) - 3)

    if output_interval is None and fmi_type == 'CoSimulation' and experiment is not None and experiment.stepSize is not None:
        output_interval = experiment.stepSize
        while (stop_time - start_time) / output_interval > 1000:
            output_interval *= 2

    if os.path.isfile(os.path.join(filename, 'modelDescription.xml')):
        unzipdir = filename
        tempdir = None
    else:
        tempdir = extract(filename)
        unzipdir = tempdir

    if use_remoting:
        # start 32-bit server
        from subprocess import Popen
        server_path = os.path.dirname(__file__)
        server_path = os.path.join(server_path, 'remoting', 'server.exe')
        if fmi_type == 'ModelExchange':
            model_identifier = model_description.modelExchange.modelIdentifier
        else:
            model_identifier = model_description.coSimulation.modelIdentifier
        dll_path = os.path.join(unzipdir, 'binaries', 'win32', model_identifier + '.dll')
        server = Popen([server_path, dll_path])
    else:
        server = None

    if fmu_instance is None:
        fmu = instantiate_fmu(unzipdir, model_description, fmi_type, visible, debug_logging, logger, fmi_call_logger, use_remoting)
    else:
        fmu = fmu_instance

    # simulate_fmu the FMU
    if fmi_type == 'ModelExchange':
        result = simulateME(model_description, fmu, start_time, stop_time, solver, step_size, relative_tolerance, start_values, apply_default_start_values, input, output, output_interval, record_events, timeout, step_finished)
    elif fmi_type == 'CoSimulation':
        result = simulateCS(model_description, fmu, start_time, stop_time, relative_tolerance, start_values, apply_default_start_values, input, output, output_interval, timeout, step_finished)

    if fmu_instance is None:
        fmu.freeInstance()

    if server is not None:
        server.kill()

    # clean up
    if tempdir is not None:
        shutil.rmtree(tempdir, ignore_errors=True)

    return result


def simulateME(model_description, fmu, start_time, stop_time, solver_name, step_size, relative_tolerance, start_values, apply_default_start_values, input_signals, output, output_interval, record_events, timeout, step_finished):

    if relative_tolerance is None:
        relative_tolerance = 1e-5

    if output_interval is None:
        if step_size is None:
            output_interval = auto_interval(stop_time - start_time)
        else:
            output_interval = step_size
            while (stop_time - start_time) / output_interval > 1000:
                output_interval *= 2

    if step_size is None:
        step_size = output_interval
        max_step = (stop_time - start_time) / 1000
        while step_size > max_step:
            step_size /= 2

    sim_start = current_time()

    time = start_time

    is_fmi1 = model_description.fmiVersion == '1.0'
    is_fmi2 = model_description.fmiVersion == '2.0'
    is_fmi3 = model_description.fmiVersion.startswith('3.0')

    # if is_fmi1:
        # fmu.setTime(time)
    # elif is_fmi2:
        # fmu.setupExperiment(startTime=start_time)

    input = Input(fmu, model_description, input_signals)

    apply_start_values(fmu, model_description, start_values, apply_default_start_values)

    # initialize
    if is_fmi1:

        input.apply(time)

        (iterationConverged,
         stateValueReferencesChanged,
         stateValuesChanged,
         terminateSimulation,
         nextEventTimeDefined,
         nextEventTime) = fmu.initialize()

        if terminateSimulation:
            raise Exception('Model requested termination during initial event update.')

    elif is_fmi2:

        # fmu.enterInitializationMode()
        input.apply(time)
        # fmu.exitInitializationMode()

        newDiscreteStatesNeeded = True
        terminateSimulation = False

        while newDiscreteStatesNeeded and not terminateSimulation:
            # update discrete states
            (newDiscreteStatesNeeded,
             terminateSimulation,
             nominalsOfContinuousStatesChanged,
             valuesOfContinuousStatesChanged,
             nextEventTimeDefined,
             nextEventTime) = fmu.newDiscreteStates()

        if terminateSimulation:
            raise Exception('Model requested termination during initial event update.')

        # fmu.enterContinuousTimeMode()

    elif is_fmi3:

        fmu.enterInitializationMode(startTime=start_time)
        input.apply(time)
        fmu.exitInitializationMode()

        newDiscreteStatesNeeded = True
        terminateSimulation = False

        while newDiscreteStatesNeeded and not terminateSimulation:
            # update discrete states
            (newDiscreteStatesNeeded,
             terminateSimulation,
             nominalsOfContinuousStatesChanged,
             valuesOfContinuousStatesChanged,
             nextEventTimeDefined,
             nextEventTime) = fmu.newDiscreteStates()

        fmu.enterContinuousTimeMode()

    # common solver constructor arguments
    solver_args = {
        'nx': model_description.numberOfContinuousStates,
        'nz': model_description.numberOfEventIndicators,
        'get_x': fmu.getContinuousStates,
        'set_x': fmu.setContinuousStates,
        'get_dx': fmu.getDerivatives,
        'get_z': fmu.getEventIndicators,
        'input': input
    }

    # select the solver
    if solver_name == 'Euler':
        solver = ForwardEuler(**solver_args)
        fixed_step = True
    elif solver_name is None or solver_name == 'CVode':
        from fmpy.sundials import CVodeSolver
        solver = CVodeSolver(set_time=fmu.setTime,
                             startTime=start_time,
                             maxStep=(stop_time - start_time) / 50.,
                             relativeTolerance=relative_tolerance,
                             **solver_args)
        step_size = output_interval
        fixed_step = False
    else:
        raise Exception("Unknown solver: %s. Must be one of 'Euler' or 'CVode'." % solver_name)

    # check step size
    if fixed_step and not np.isclose(round(output_interval / step_size) * step_size, output_interval):
        raise Exception("output_interval must be a multiple of step_size for fixed step solvers")

    recorder = Recorder(fmu=fmu,
                        modelDescription=model_description,
                        variableNames=output,
                        interval=output_interval)

    # record the values for time == start_time
    recorder.sample(time)

    t_next = start_time

    # simulation loop
    while time < stop_time:

        if timeout is not None and (current_time() - sim_start) > timeout:
            break

        if fixed_step:
            if time + step_size < stop_time + eps:
                t_next = time + step_size
            else:
                break
        else:
            if time + eps >= t_next:  # t_next has been reached
                # integrate to the next grid point
                t_next = np.floor(time / output_interval) * output_interval + output_interval
                if t_next < time + eps:
                    t_next += output_interval

        # get the next input event
        t_input_event = input.nextEvent(time)

        # check for input event
        input_event = t_input_event <= t_next

        if input_event:
            t_next = t_input_event

        time_event = nextEventTimeDefined and nextEventTime <= t_next

        if time_event and not fixed_step:
            t_next = nextEventTime

        if t_next - time > eps:
            # do one step
            state_event, roots_found, time = solver.step(time, t_next)
        else:
            # skip
            time = t_next

        # set the time
        fmu.setTime(time)

        # apply continuous inputs
        input.apply(time, discrete=False)

        # check for step event, e.g.dynamic state selection
        if is_fmi1:
            step_event = fmu.completedIntegratorStep()
        else:
            step_event, _ = fmu.completedIntegratorStep()
            step_event = step_event != fmi2False

        # handle events
        if input_event or time_event or state_event or step_event:

            if record_events:
                # record the values before the event
                recorder.sample(time, force=True)

            if is_fmi1:

                if input_event:
                    input.apply(time=time, after_event=True)

                iterationConverged = False

                # update discrete states
                while not iterationConverged and not terminateSimulation:
                    (iterationConverged,
                     stateValueReferencesChanged,
                     stateValuesChanged,
                     terminateSimulation,
                     nextEventTimeDefined,
                     nextEventTime) = fmu.eventUpdate()

                if terminateSimulation:
                    break

            elif is_fmi2:

                fmu.enterEventMode()

                if input_event:
                    input.apply(time=time, after_event=True)

                newDiscreteStatesNeeded = True

                # update discrete states
                while newDiscreteStatesNeeded and not terminateSimulation:
                    (newDiscreteStatesNeeded,
                     terminateSimulation,
                     nominalsOfContinuousStatesChanged,
                     valuesOfContinuousStatesChanged,
                     nextEventTimeDefined,
                     nextEventTime) = fmu.newDiscreteStates()

                if terminateSimulation:
                    break

                fmu.enterContinuousTimeMode()

            else:

                fmu.enterEventMode(stepEvent=step_event, rootsFound=roots_found, timeEvent=time_event)

                if input_event:
                    input.apply(time=time, after_event=True)

                newDiscreteStatesNeeded = True

                # update discrete states
                while newDiscreteStatesNeeded and not terminateSimulation:
                    (newDiscreteStatesNeeded,
                     terminateSimulation,
                     nominalsOfContinuousStatesChanged,
                     valuesOfContinuousStatesChanged,
                     nextEventTimeDefined,
                     nextEventTime) = fmu.newDiscreteStates()

                if terminateSimulation:
                    break

                fmu.enterContinuousTimeMode()

            solver.reset(time)

            if record_events:
                # record values after the event
                recorder.sample(time, force=True)

        if abs(time - round(time / output_interval) * output_interval) < eps and time > recorder.lastSampleTime + eps:
            # record values for this step
            recorder.sample(time, force=True)

        if step_finished is not None and not step_finished(time, recorder):
            break

#    fmu.terminate()

    del solver

    return recorder.result()

