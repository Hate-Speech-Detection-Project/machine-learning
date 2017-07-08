function submitComment() {
    const comment = document.getElementById("comment").value;
    const url = document.getElementById("url").value;

    var xmlHttp = new XMLHttpRequest();
    xmlHttp.open( "POST", url, false );

    xmlHttp.onreadystatechange = function() {
        if (xmlHttp.readyState == 4 && xmlHttp.status == 200){
           console.log("fine");
        }
    }
    xmlHttp.setRequestHeader('Content-type', 'application/json');

    const payload = {
            'comment': comment,
            'url': url
    }


    xmlHttp.send(JSON.stringify(payload));


}
