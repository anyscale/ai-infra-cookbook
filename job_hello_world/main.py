import os
import ray


@ray.remote
def f(i):
    # This print statement is running in a separate worker process.
    print(f"The value of EXAMPLE_ENV_VAR is {os.environ['EXAMPLE_ENV_VAR']}.")
    return i ** 2


# Execute 100 tasks across the cluster.
results = ray.get([f.remote(i) for i in range(100)])
print(results)