# segmentation/model_training.py
# Train the segmentation model.

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

def compile_and_train_model(model, X_train, y_train, X_valid, y_valid, optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'], batch_size=32, epochs=10, callbacks=None):
    """Compile and train the segmentation model."""
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    # Define default callbacks if not provided
    if callbacks is None:
        callbacks = [
            EarlyStopping(patience=10, verbose=1),
            ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
            ModelCheckpoint('path_to_seg_model_checkpoint', verbose=1, save_best_only=True, save_weights_only=True)
        ]
    
    # Train the model
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks, validation_data=(X_valid, y_valid))
    
    return history
