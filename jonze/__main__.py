import os
import tensorflow as tf
import numpy as np

from . import SlotGatedModel

from . import utils

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

configp = ConfigProto()
configp.gpu_options.allow_growth = True
session = InteractiveSession(config=configp)

DATASETS_ROOT = "datasets/"
MODELS_ROOT = "models/"

TRAIN_FOLDER_NAME = "train"
TEST_FOLDER_NAME = "test"
VALID_FOLDER_NAME = "valid"

TEXT_VOCAB_FILENAME = 'in_vocab'
SLOTS_VOCAB_FILENAME = 'slot_vocab'
INTENTS_VOCAB_FILENAME = 'intent_vocab'

TEXT_FILENAME = 'seq.in'
SLOTS_FILENAME = 'seq.out'
INTENTS_FILENAME = 'label'

CHECKPOINTS_FOLDER_NAME = "checkpoints"

MAX_CHECKPOINTS_TO_KEEP = 3

def train(dataset: str, number_of_epochs: int = 50, allow_early_stop: bool = False,
          datasets_root: str = DATASETS_ROOT, models_root: str = MODELS_ROOT, layer_size: int = 128, batch_size: int = 64, patience: int = 5):
    #tf.keras.backend.clear_session()

    full_train_path = os.path.join(datasets_root, dataset, TRAIN_FOLDER_NAME)
    full_test_path = os.path.join(datasets_root, dataset, TEST_FOLDER_NAME)
    full_valid_path = os.path.join(datasets_root, dataset, VALID_FOLDER_NAME)

    in_vocab = utils.loadVocabulary(os.path.join(datasets_root, dataset, 'in_vocab'))
    slot_vocab = utils.loadVocabulary(os.path.join(datasets_root, dataset, 'slot_vocab'))
    intent_vocab = utils.loadVocabulary(os.path.join(datasets_root, dataset, 'intent_vocab'))

    model = SlotGatedModel.SlotGatedModel(len(in_vocab['vocab']), len(slot_vocab['vocab']), len(intent_vocab['vocab']), layer_size = layer_size)

    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    ckpt = tf.train.Checkpoint(step=tf.Variable(-1), optimizer=opt, net=model)
    manager = tf.train.CheckpointManager(ckpt, os.path.join(models_root, dataset, CHECKPOINTS_FOLDER_NAME), max_to_keep=MAX_CHECKPOINTS_TO_KEEP)
    data_processor = None
    valid_err = 0
    no_improve= 0
    save_path = os.path.join(models_root, dataset)
    for epoch in range(number_of_epochs):
        while True:
            if data_processor == None:
                i_loss = 0
                s_loss = 0
                batches = 0
                data_processor = utils.DataProcessor(os.path.join(full_train_path, TEXT_FILENAME), os.path.join(full_train_path, SLOTS_FILENAME), os.path.join(full_train_path, INTENTS_FILENAME), in_vocab, slot_vocab, intent_vocab)
            in_data, slot_labels, slot_weights, length, intent_labels,in_seq,_,_ = data_processor.get_batch(batch_size)
        
            with tf.GradientTape() as tape:
                slots, intent = model(in_data, length, isTraining = True)
                intent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=intent_labels, logits=intent)
                #slot_loss
                slots_out = tf.reshape(slots, [-1,len(slot_vocab['vocab'])])
                slots_shape = tf.shape(slot_labels)
                slot_reshape = tf.reshape(slot_labels, [-1])
                crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=slot_reshape, logits=slots_out)
                crossent = tf.reshape(crossent, slots_shape)
                slot_loss = tf.reduce_sum(crossent*slot_weights, 1)
                total_size = tf.reduce_sum(slot_weights, 1)
                total_size += 1e-12
                slot_loss = slot_loss / total_size
          
                total_loss = intent_loss + slot_loss + tf.reduce_sum(model.losses)
        
            grads = tape.gradient(total_loss, model.trainable_weights)
            opt.apply_gradients(zip(grads, model.trainable_weights))
            s_loss = s_loss + tf.reduce_sum(slot_loss)/tf.cast(batch_size, tf.float32)
            i_loss = i_loss +  tf.reduce_sum(intent_loss)/tf.cast(batch_size, tf.float32)
            batches = batches + 1
            #print(data_processor.)
            if data_processor.end == 1:
                data_processor.close()
                data_processor = None
                break
            
        #print("Training Epoch: " ,epoch," Slot Loss: ",s_loss/batches, " Intent_Loss: ", i_loss/batches)
        print("EPOCH: ", epoch, " *******************************************************************")
        print('Train:', end="\t")
        _ = utils.validate(os.path.join(full_train_path, TEXT_FILENAME), os.path.join(full_train_path, SLOTS_FILENAME), os.path.join(full_train_path, INTENTS_FILENAME), in_vocab, slot_vocab, intent_vocab, model, batch_size)
    
        print('Valid:', end="\t")
        epoch_valid_slot, epoch_valid_intent, epoch_valid_err,valid_pred_intent,valid_correct_intent,valid_pred_slot,valid_correct_slot,valid_words = utils.validate(os.path.join(full_valid_path, TEXT_FILENAME), os.path.join(full_valid_path, SLOTS_FILENAME), os.path.join(full_valid_path, INTENTS_FILENAME), in_vocab, slot_vocab, intent_vocab, model, batch_size)

        print('Test:', end="\t")
        epoch_test_slot, epoch_test_intent, epoch_test_err,test_pred_intent,test_correct_intent,test_pred_slot,test_correct_slot,test_words = utils.validate(os.path.join(full_test_path, TEXT_FILENAME), os.path.join(full_test_path, SLOTS_FILENAME), os.path.join(full_test_path, INTENTS_FILENAME), in_vocab, slot_vocab, intent_vocab, model, batch_size)
    
        ckpt.step.assign_add(1)
        if epoch_valid_err <= valid_err:
            no_improve += 1
        else:
            valid_err = epoch_valid_err
            no_improve = 0
            print("Saving", str(ckpt.step), "with valid accuracy:", valid_err   )
            manager.save()

        if allow_early_stop == True:
            if no_improve > patience:
                print("EARLY BREAK")
                break

    manager.save()
    model.summary()

def test(dataset: str, batch_size: int, datasets_root: str = DATASETS_ROOT, models_root: str = MODELS_ROOT, layer_size: int = 128):
    full_test_path = os.path.join(datasets_root, dataset, TEST_FOLDER_NAME)

    in_vocab = utils.loadVocabulary(os.path.join(datasets_root, dataset, 'in_vocab'))
    slot_vocab = utils.loadVocabulary(os.path.join(datasets_root, dataset, 'slot_vocab'))
    intent_vocab = utils.loadVocabulary(os.path.join(datasets_root, dataset, 'intent_vocab'))

    #Let'stry to clear slate and reload maodel  .....
    import time
    tf.keras.backend.clear_session()

    model = SlotGatedModel.SlotGatedModel(len(in_vocab['vocab']), len(slot_vocab['vocab']), len(intent_vocab['vocab']), layer_size=layer_size)
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

    ckpt = tf.train.Checkpoint(step=tf.Variable(-1), optimizer=opt, net=model)
    manager = tf.train.CheckpointManager(ckpt, os.path.join(models_root, dataset, CHECKPOINTS_FOLDER_NAME), max_to_keep=MAX_CHECKPOINTS_TO_KEEP)
    ckpt.restore(manager.latest_checkpoint)

    data_processor_test = utils.DataProcessor(os.path.join(full_test_path, TEXT_FILENAME), os.path.join(full_test_path, SLOTS_FILENAME), os.path.join(full_test_path, INTENTS_FILENAME), in_vocab, slot_vocab, intent_vocab)
    in_data, slot_labels, slot_weights, length, intent_labels,in_seq,_,_ = data_processor_test.get_batch(batch_size)
    data_processor_test.close()

    t = time.perf_counter()
    slots, intent = model(in_data, length, isTraining = False)
    elapsed = time.perf_counter() - t
    print("Milli seconds per query:", (elapsed*1000)/float(100.0))

    pred_intents = []
    correct_intents = []
    slot_outputs = []
    correct_slots = []
    input_words = []

    for i in np.array(intent):
        pred_intents.append(np.argmax(i))
    for i in intent_labels:
        correct_intents.append(i)

    pred_slots = slots
    for p, t, i, l in zip(pred_slots, slot_labels, in_data, length):
        p = np.argmax(p, 1)
        tmp_pred = []
        tmp_correct = []
        tmp_input = []
        for j in range(l):
            tmp_pred.append(slot_vocab['rev'][p[j]])
            tmp_correct.append(slot_vocab['rev'][t[j]])
            tmp_input.append(in_vocab['rev'][i[j]])

        slot_outputs.append(tmp_pred)
        correct_slots.append(tmp_correct)
        input_words.append(tmp_input)

    pred_intents = np.array(pred_intents)
    correct_intents = np.array(correct_intents)
    accuracy = (pred_intents==correct_intents)
    semantic_error = accuracy
    accuracy = accuracy.astype(float)
    accuracy = np.mean(accuracy)*100.0

    index = 0
    for t, p in zip(correct_slots, slot_outputs):
        # Process Semantic Error
        if len(t) != len(p):
            raise ValueError('Error!!')

        for j in range(len(t)):
            if p[j] != t[j]:
                semantic_error[index] = False
                break
        index += 1
    semantic_error = semantic_error.astype(float)
    semantic_error = np.mean(semantic_error)*100.0

    f1, precision, recall = utils.computeF1Score(correct_slots, slot_outputs)
    print('slot f1: ' + str(f1) + '\tintent accuracy: ' + str(accuracy) + '\tsemantic_accuracy: ' + str(semantic_error))

if __name__ == "__main__":
    tf.keras.backend.clear_session()
    train(dataset="itmo", layer_size=12)
    test(dataset="itmo", layer_size=12, batch_size=46)