function submitComment(event) {
    event.preventDefault()
    const comment = document.getElementById("comment").value;
    const url = document.getElementById("url").value;
    const apiURL = '//127.0.0.1:5000/predict';

    var xmlHttp = new XMLHttpRequest();
    xmlHttp.open( "POST", apiURL, false );

    xmlHttp.onreadystatechange = function() {
        if (xmlHttp.readyState == 4 && xmlHttp.status == 200){
           alert(JSON.stringify(JSON.parse(xmlHttp.responseText)))
        }
    }
    xmlHttp.setRequestHeader('Content-type', 'application/json');
    xmlHttp.setRequestHeader('Access-Control-Allow-Origin', '*');

    const payload = {
            'comment': comment,
            'url': url
    }


    xmlHttp.send(JSON.stringify(payload));


}
