# Submission 2: Hotel-Reservations-Prediction
Nama: Axel Sean Cahyono Putra

Username dicoding: axelseancp

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [Hotel_Reservations_Dataset](https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset) |
| Masalah | Reservasi hotel di jaman sekarang dapat dilakukan dengan mudah secara online melalui aplikasi booking hotel seperti Traveloka, Trivago, Agoda, dll. Namun karena kemudahan tersebut muncullah masalah baru yaitu pembatalan reservasi hotel, entah karena ada rencana dadakan, jadwalnya bertabrakan, dll. Efek dari pembatalan reservasi tidak terlalu berpengaruh pada customer, namun sangat berpengaruh pada hotel karena bisa menurunkan pendapatan hotel. Contohnya jika waktu liburan ada customer yang memesan kamar terakhir lalu banyak customer datang ingin booking tapi tidak bisa karena sudah penuh, kemudian customer yang memesan kamar tersebut membatalkan pesanannya, hal ini berpengaruh karena hotel bisa saja memberikan kamar itu untuk orang lain jika sudah tahu dari awal. |
| Solusi machine learning | Untuk membantu hotel dalam mengatasi masalah ini maka dapat menggunakan sistem **Hotel-Reservations-Prediction** yang diharapkan dapat memprediksi apakah suatu customer akan melakukan pembatalan reservasi atau tidak. |
| Metode pengolahan |  |
| Arsitektur model |  |
| Metrik evaluasi | Metrik yang digunakan adalah **BinaryAccuracy** karena *target_variable* dari dataset ini bersifat binary (dibatalkan, tidak dibatalkan) |
| Performa model | Model yang dijalankan memiliki akurasi dan loss yang stabil, serta tidak menunjukkan tanda tanda *overfitting*. Model ini dapat digunakan untuk melakukan *request* prediksi. |
| Opsi deployment | Model ini dideploy menggunakan platform railway |
| Web app | Tautan web app yang digunakan untuk mengakses model serving. Contoh: [hr-model](https://hotel-reservations-prediction-production.up.railway.app/v1/models/hr-model/metadata)|
| Monitoring | Monitoring model ini dilakukan menggunakan prometheus. Monitoring yang dilakukan adalah memantau jumlah request ke dalam model. |