from question_classification.data.qc_dataloader import QCDataLoader
from question_classification.models.hier_qc import HierRNN
import torch

def main():
    torch.set_printoptions(edgeitems=100000)
    DL = QCDataLoader('WordTokenizer')
    print(DL.tokenizer.vocab_size)
    model = HierRNN(DL.tokenizer.vocab_size, 20, 20, 20, 10, DL.max_word_length, DL.max_sen_length)
    # print('Train')
    # for x,y in DL.train_loader:
    #     print(model.forward(x))
    #     break



    print('----')
    # DL2 = QCDataLoader('CharacterTokenizer')

if __name__ == '__main__':
    main()
