let iterationFinished = false;
let ws;


function setWithDataURL(url, elem) {
    let isFirefox = navigator.userAgent.search("Firefox") > -1;
    let img = $("<img>");
    img.attr("class", "replace");
    img[0].onload = () => {
        let h = img[0].naturalHeight;
        let w = img[0].naturalWidth;
        let scale = parseInt($(elem).css("width")) / Math.max(h, w);
        img.attr("height", Math.min(h, h * scale));
        img.attr("width", Math.min(w, w * scale));
        if (!isFirefox) $(`#${elem.id} .replace`).replaceWith(img);
    };
    img.attr("src", url);
    if (isFirefox) $(`#${elem.id} .replace`).replaceWith(img);
}


function wsConnect() {
    ws = new WebSocket("ws://" + window.location.host + "/websocket");

    ws.onclose = () => {
        if (!iterationFinished) {
            let status = $("#status")[0];
            status.innerText = "Disconnected from the backend.";
            status.style.display = "";
        }
    };

    ws.onerror = ws.onclose;

    ws.onmessage = (e) => {
        let msg = JSON.parse(e.data);

        switch (msg._type) {
        case "Iterate":
            setWithDataURL(msg.image, $("#image")[0]);
            $("#step")[0].innerText = msg.step;
            $("#steps")[0].innerText = msg.steps;
            $("#time")[0].innerText = msg.time.toFixed(2);
            $("#update-size")[0].innerText = msg.update_size.toFixed(2);
            $("#loss")[0].innerText = msg.loss.toExponential(6);
            $("#tv")[0].innerText = msg.tv.toFixed(2);
            $("#status")[0].style.display = "none";
            break;

        case "IterationFinished":
            iterationFinished = true;
            $("#status")[0].innerText = "Iteration finished.";
            $("#status")[0].style.display = "";
            break;
        }
    };

    ws.onopen = () => {
        let status = $("#status")[0];
        status.innerText = "Waiting for the first iteration...";
    };
}


$(document).ready(() => {
    wsConnect();
});
