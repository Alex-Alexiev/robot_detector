import xml_to_csv
import generate_tfrecord

def main():
    xml_df = xml_to_csv.xml_to_csv('../images/train')
    xml_df.to_csv('../images/train_labels.csv', index=None)
    xml_df = xml_to_csv.xml_to_csv('../images/test')
    xml_df.to_csv('../images/test_labels.csv', index=None)

    generate_tfrecord.generate_tfrecord('../images/train_labels.csv', '../training/train.record', '../images/train')
    generate_tfrecord.generate_tfrecord('../images/test_labels.csv', '../training/test.record', '../images/test')

if __name__ == "__main__":
    main()