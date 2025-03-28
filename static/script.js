$(document).ready(function () {
    $("#currency-form").submit(function (event) {
        event.preventDefault();

        let amount = $("#amount").val();
        let from = $("#from").val();
        let to = $("#to").val();
        let question = $("#question").val();

        $.ajax({
            url: "/convert",
            method: "POST",
            contentType: "application/json",
            data: JSON.stringify({ amount: amount, from: from, to: to, question: question }),
            success: function (response) {
                let chatEntry = `<div class="question">💬 <strong>Question:</strong> ${question}</div>
                                 <div class="response">🔹 <strong>Response:</strong> ${response.result}</div>`;
                $("#chat-history").prepend(chatEntry);
            },
            error: function () {
                alert("Error in conversion. Please try again.");
            }
        });
    });

    $("#prediction-form").submit(function (event) {
        event.preventDefault();

        let from = $("#from").val();
        let to = $("#to").val();
        let days = $("#days_ahead").val();

        $.ajax({
            url: "/predict",
            method: "POST",
            contentType: "application/json",
            data: JSON.stringify({ from: from, to: to, days: days }),
            success: function (response) {
                let chatEntry = `<div class="response">🔮 <strong>Prediction:</strong> ${response.result}</div>`;
                $("#chat-history").prepend(chatEntry);
            },
            error: function () {
                alert("Error in prediction. Please try again.");
            }
        });
    });
});
