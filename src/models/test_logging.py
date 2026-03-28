import logging
import time

logging.basicConfig(filename="training_log.txt", level=logging.INFO)

num_epochs = 500
for epoch in range(1, num_epochs + 1):

    time.sleep(1.0)

    training_loss = 0.114
    validation_accuracy = 0.514

    logging.info(f'Epoch {epoch}/{num_epochs}, Training Loss: {training_loss:.4f}, Validation Accuracy: {validation_accuracy:.4f}')
