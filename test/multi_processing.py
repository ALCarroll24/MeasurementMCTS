import time
from multiprocessing import Pool

# Example computationally expensive function
def expensive_function(x):
    time.sleep(0.1)  # Simulate a time-consuming operation
    return x * x

# Serial execution (without multiprocessing)
def process_serial(numbers):
    result = []
    for number in numbers:
        result.append(expensive_function(number))
    return result

# Parallel execution (with multiprocessing)
def process_parallel(numbers):
    with Pool(5) as pool:
        result = pool.map(expensive_function, numbers)
    return result

if __name__ == "__main__":
    numbers = list(range(20))  # Example list of numbers

    # Serial execution
    start_time = time.time()
    result_serial = process_serial(numbers)
    end_time = time.time()
    serial_duration = end_time - start_time
    print(f"Serial execution took {serial_duration:.2f} seconds")

    # Parallel execution
    start_time = time.time()
    result_parallel = process_parallel(numbers)
    end_time = time.time()
    parallel_duration = end_time - start_time
    print(f"Parallel execution took {parallel_duration:.2f} seconds")

    # Speedup
    speedup = serial_duration / parallel_duration
    print(f"Speedup: {speedup:.2f}x")
