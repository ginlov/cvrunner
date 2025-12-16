import numpy as np

from datetime import datetime
from cvrunner.utils.logger import get_cv_logger

def generate_dummy_images(num_images=5, height=64, width=64):
    images = []
    for _ in range(num_images):
        img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        images.append(img)
    return images

if __name__ == "__main__":
    logger = get_cv_logger()
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.init_wandb(project="test_logger", run_name=f"test_run_{now}", config={"test_param": 42})
    logger.info("This is an info message from test_logger.py")
    logger.debug("This is a debug message from test_logger.py")
    logger.warning("This is a warning message from test_logger.py")
    logger.error("This is an error message from test_logger.py")
    logger.critical("This is a critical message from test_logger.py")

    # This should log 25 images in total
    for i in range(5):
        logger.log_images(image_ids=[i*5 + j for j in range(5)],
                images=generate_dummy_images(), step=0)

    # This should log 25 images in total
    for i in range(5):
        logger.log_images(image_ids=[i*5 + j for j in range(5)],
                images=generate_dummy_images(), local_step=10)

    logger.log_histogram("test_histogram", np.random.randn(1000), local_step=5)
    logger.log_histogram("test_histogram", np.random.randn(1000), local_step=10)
