$(document).ready(function () {
    // Init
    // $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();
    $('#result-details').hide();

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);            
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        $('#result-details').text('');
        $('#result-details').hide();
        readURL(this);
    });

    // Predict
    function getPrediction() {

        var imageInput = $('#imagePreview').attr('style').split(",")[1];

        var base64ImageData = imageInput.substring(0,imageInput.length-3);
        
        $(this).hide();
        $('.loader').show();

        fetch("/predict",{
            method: "POST",
            body: JSON.stringify({image:base64ImageData}),
            headers: {
                'Content-Type': 'application/json'
                // 'Content-Type': 'application/x-www-form-urlencoded',
            },
        })
        .then(data => data.text())
        .then(data => console.log(data));

        $(this).show();
        $('.loader').hide();
    }

    $('#btn-predict').click(getPrediction);

    // $('#btn-predict').click(function () {
        
    //     var form_data = new FormData($('#upload-file')[0]);



    //     // Show loading animation
    //     $(this).hide();
    //     $('.loader').show();

        


    //     // Make prediction by calling api /predict
    //     $.ajax({
    //         type: 'POST',
    //         url: '/predict',
    //         data: fd,
    //         contentType: false,
    //         cache: false,
    //         processData: false,
    //         async: true,
    //         success: function (data) {
    //             // Get and display the result
    //             $('.loader').hide();
    //             $('#result').fadeIn(600);
    //             $('#result').text(' Result:  ' + data);
    //             //$('#result-details').fadeIn(600);
    //             //$('#result-details').text(data);
    //             console.log('Success!');
    //         },
    //     });
    // });

});