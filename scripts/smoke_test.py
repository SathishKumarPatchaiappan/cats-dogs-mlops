import requests


def run_smoke_test():
    health_response = requests.get(
        "http://127.0.0.1:8000/health",
        timeout=10
    )

    if health_response.status_code != 200:
        raise Exception("Health check failed")

    print("Health check passed")

    print(health_response.json())


if __name__ == "__main__":
    run_smoke_test()