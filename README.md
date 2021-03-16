# THE SIMPSONS CHARACTER RECOGNITION APPLICATION

Members: Daniel Fischer, Mo Habib, Fazliddin Naimov, Kevin Freehill, Dahmane Skendraoui, Peter Kim

[Final Presentation](Presentation%20for%20Final%20Project.pdf)

[Visit Deployed Page](http://simpsons-classification.herokuapp.com/)

## ABOUT

Training a convolutional neural network to recognize The Simpsons characters. Our approach to solve this problem will be based on convolutional neural networks (CNNs): multi-layered feed-forward neural networking able to learn many features.

Technology Stack Used:
- Python
- HTML/CSS/Bootstrap/Javascript
- Keras
- Tensorflow
- Flask

## PROCESS
### CNN Model
#### Dataset

We used kaggle Simpsons dataset that has more than 40 characters of pictures. For training, we only used characters that have more than 290 images.

```python
characters = [k.split('/')[2] for k in glob.glob('./characters/*') if len([p for p in glob.glob(k+'/*') if 'edited' in p or 'pic_vid' in p]) > 290]
```

The model was trained to classify 18 characters only; here is the list:
1. Abraham Grampa
2. Apu Nahasapeemapetilon
3. Bart Simpson
4. Charles Montgomery Burns
5. Chief Wiggum
6. Comic Book Guy
7. Edna Krabappel
8. Homer Simpson
9. Kent Brockman
10. Krusty the Clown
11. Lisa Simpson
12. Marge Simpson
13. Milhouse van Houten
14. Moe Szyslak
15. Ned Flanders
16. Nelson Muntz
17. Principal Skinner
18. Sideshow Bob

[Dataset Link](https://www.kaggle.com/alexattia/the-simpsons-characters-dataset)

#### Training
Splitting the data to Train and Test using get_dataset function from train.py.

```python
imp.reload(train)
X_train, X_test, y_train, y_test = train.get_dataset(save=True)
```

We used a feed-forward 4 convolutional layers with ReLU activation followed by a fully connected hidden layer. The model iterated over batches of the training set (batch size: 32) for 200 epochs. We also used data augmentation that did several random variations over the pictures, so the model never sees the same picture twice. This helps prevent overfitting and allows the model to generalize better.

```python
datagen = ImageDataGenerator(
 featurewise_center=False, # set input mean to 0 over the dataset
 samplewise_center=False, # set each sample mean to 0
 featurewise_std_normalization=False, # divide inputs by std 
 samplewise_std_normalization=False, # divide each input by its std
 rotation_range=0, # randomly rotate images in the range 
 width_shift_range=0.1, # randomly shift images horizontally 
 height_shift_range=0.1, # randomly shift images vertically 
 horizontal_flip=True, # randomly flip images
 vertical_flip=False) # randomly flip images
 ```
Loss and Accuracy (Validation and Training) during training
![Loss and Accuracy (Validation and Training) during training](images/loss.png)

#### Classification evaluation
The accuracy (f1-score) is really good: above 90 % for every character except Lisa. The precision for Lisa is 82%. Maybe Lisa is mixed up with other characters.

![4 convolutional layers net](images/4cln.png)

![Confusion Matrix](images/confusion.png)

#### Improving the CNN model

To make the neural net understand more details and more complexity, we can add more convolutional layers. We tried with 6 convolutional layers and going deeper (dimensions of the output space 32, 64, 512 vs. 32, 64, 256, 1024). It has improved the accuracy (precision and recall), as you can see below. The lower precision is 0.89 for Nelson Muntz, and we only had 300 training examples for this character. Moreover, this model converges quicker: only 40 epochs (vs. 200).

![6 convolutional layers net](images/6cln.png)

#### Visualizing predicted characters

![Actual and predicted characters for 12 different characters](images/actvspred.png)

#### Predict from file and URL
Created two functions that uses model to predict from image and URL.

```python
def file_predict(image_path, all_perc=False):
    image = cv2.imread(image_path)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()
    pic = cv2.resize(image, (64,64))
    a = model.predict_proba(pic.reshape(1, 64, 64,3))[0]
    if all_perc:
        print('\n'.join(['{} : {}%'.format(map_characters[i], round(k*100)) for i,k in sorted(enumerate(a), key=lambda x:x[1], reverse=True)]))
    else:
        return map_characters[np.argmax(a)].replace('_',' ').title()
def url_predict(url, all_perc=False):
    image = url_to_image(url)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()
    pic = cv2.resize(image, (64,64))
    a = model.predict_proba(pic.reshape(1, 64, 64,3))[0]
    if all_perc:
        print('\n'.join(['{} : {}%'.format(map_characters[i], round(k*100)) for i,k in sorted(enumerate(a), key=lambda x:x[1], reverse=True)]))
    else:
        return map_characters[np.argmax(a)].replace('_',' ').title()
```
![Predict from file](images/file_pred.jpg)
![Predict from URL](images/url_pred.jpg)

### Flask

We used a Javascript/HTML frontend with a Flask backend server written in python. The back end comprises two endpoints, as displayed below - the first endpoint renders the page while the second handle prediction requests sent from the frontend. All requests include a base64 string representation of a picture file that is then decoded and converted into a NumPy array to pass through the model. Predictions passed back from the model are then relayed to the front end as a string, thus completing the initial request.

```python
@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
```

### HTML/CSS/JS

The HTML page is comprised of two buttons. JavaScript will save the file and transport it to the server for prediction.

```javascript
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                $('.output-container').show();
                $('#result').fadeIn(600);
                $('#result').text(' This is ' + data);
                console.log(data)
            },
        });
    });
```

## FINAL APPLICATION
![screenshot](images/app.png)
