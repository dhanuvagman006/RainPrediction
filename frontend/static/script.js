document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("weather-form");
    const resultContainer = document.getElementById("prediction-result");
    const loadingIndicator = document.getElementById("loading");

    form.addEventListener("submit", async function (event) {
        event.preventDefault();
        resultContainer.innerHTML = "";
        loadingIndicator.style.display = "block";

        const formData = new FormData(form);
        const jsonData = {};
        formData.forEach((value, key) => {
            jsonData[key] = parseFloat(value) || 0;
        });

        try {
            const response = await fetch("http://127.0.0.1:5000/predict", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(jsonData)
            });

            const data = await response.json();
            loadingIndicator.style.display = "none";
            
            if (data.prediction) {
                resultContainer.style.display="block";
                resultContainer.innerHTML = `
                    <div style="margin-top: 20px; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold;">
                        <span style="color: ${data.prediction === 'Rainfall' ? '#ffd5ca' : '#55ff33'}">
                            Prediction: ${data.prediction} at ${data.accuracy}% Of Accuracy. 
                        </span>
                    </div>
                `;
            } else if (data.error) {
                resultContainer.innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
            }
        } catch (error) {
            loadingIndicator.style.display = "none";
            resultContainer.innerHTML = `<p style="color: red;">Error fetching prediction: ${error.message}</p>`;
        }
    });
});
