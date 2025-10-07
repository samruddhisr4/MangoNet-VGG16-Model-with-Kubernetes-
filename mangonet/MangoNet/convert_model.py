import os
import tensorflow as tf

H5_PATH = os.path.join(os.getcwd(), 'mango_classification_model.h5')
SAVED_DIR = os.path.join(os.getcwd(), 'mango_saved_model')

def main():
    if not os.path.exists(H5_PATH):
        print('HDF5 model not found at', H5_PATH)
        return 1

    print('Loading HDF5 model from', H5_PATH)
    model = tf.keras.models.load_model(H5_PATH, compile=False)

    # Save as SavedModel directory
    if os.path.exists(SAVED_DIR):
        print('Removing existing', SAVED_DIR)
        import shutil
        shutil.rmtree(SAVED_DIR)

    print('Saving SavedModel to', SAVED_DIR)
    model.save(SAVED_DIR, save_format='tf')
    print('SavedModel written to', SAVED_DIR)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
