document.addEventListener("DOMContentLoaded", function () {
    console.log("JavaScript is working! 🎉");

    const form = document.querySelector("form");

    if (form) {
        console.log("Form found! Applying animation...");
        form.style.opacity = "0";
        form.style.transform = "translateY(20px)";

        setTimeout(() => {
            form.style.transition = "opacity 0.8s ease-in-out, transform 0.8s ease-in-out";
            form.style.opacity = "1";
            form.style.transform = "translateY(0)";
            console.log("Animation applied successfully!");
        }, 300);
    } else {
        console.error("Form not found! ❌");
    }
});


