# Submission 1: Student-Performance-Classification
Nama: Axel Sean Cahyono Putra

Username dicoding: axelseancp

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [Student_performance_data](https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset) |
| Masalah | Seseorang yang mengikuti program pendidikan di sekolah atau lembaga lainnya dapat disebut dengan murid atau pelajar. Tujuan seorang pelajar tentunya adalah untuk belajar dan menambah ilmu. Dalam hal itu diciptakan sistem ujian untuk mengetes seberapa bagus pemahaman seorang pelajar terhadap materi yang disampaikan. Namun terkadang ada pelajar yang mengalami kesulitan dalam belajar entah karena faktor lingkungan atau ketidakmampuan untuk mengikuti pelajaran. Hal ini tentunya berdampak tidak hanya kepada murid tersebut namun juga kepada guru dan orang tua. |
| Solusi machine learning | Untuk membantu murid agar lebih mudah meningkatkan performa mereka. Guru atau orang tua bisa mencari tahu langkah langkah yang dapat dilakukan untuk membantu sang murid. Dengan menggunakan sistem *Student performance classification* guru dan orang tua dapat mengetahui hal yang kurang diperhatikan. Dan menggunakan informasi tersebut diharapkan murid dapat meningkatkan performa mereka dalam mengikuti pelajaran. |
| Metode pengolahan | Pada data ini terdapat 15 feature yang mengindikasikan aktifitas atau keadaan murid, 13 feature akan digunakan untuk proses klasifikasi dengan target feature yaitu *grade class* (peringkat kelas) yang terdiri dari angka 0(A) sampai 4(F). Kemudian data akan di *split* dengan rasio 80:20 lalu dilakukan transformasi berupa *standardization* untuk feature bertipe *float*, dan menerapkan *One Hot Encoding* pada fitur categorical. Feature yang ditransformasi akan di *rename* kecuali untuk feature *One Hot Encoding*. |
| Arsitektur model | Arsitektur model yang digunakan adalah *Input layer* dengan *shape* (13,) dipadukan dengan *Dense layer* yang jumlah *layer* dan *units* diacak menggunakan *Tuner*. Output model yang dihasilkan memiliki *shape* (5,) dengan activation function *softmax* |
| Metrik evaluasi | Metrik yang digunakan adalah **Confusion Matrix** |
| Performa model | Deksripsi performa model yang dibuat |