// --- SCROLL ANIMATION & PARALLAX ---
window.addEventListener('scroll', () => {
    const scrolled = window.pageYOffset;
    const heroBg = document.querySelector('.hero-bg');
    if(heroBg) heroBg.style.transform = `translateY(${scrolled * 0.4}px)`;
    
    document.querySelectorAll('.fade-up').forEach(el => {
        const top = el.getBoundingClientRect().top;
        if (top < window.innerHeight - 100) el.classList.add('visible');
    });
});
window.dispatchEvent(new Event('scroll')); // Init

// --- APP LOGIC ---
const fileInput = document.getElementById('fileInput');

fileInput.addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (!file) return;

    // 1. Show Scanner
    const reader = new FileReader();
    reader.onload = function(e) {
        document.getElementById('scan-img').src = e.target.result;
        document.getElementById('view-upload').style.display = 'none';
        document.getElementById('view-scanner').style.display = 'block';
        
        // Scroll smoothly to scanner
        document.getElementById('app').scrollIntoView({behavior: 'smooth'});
        
        // 2. Call Backend
        uploadImage(file);
    }
    reader.readAsDataURL(file);
});

function uploadImage(file) {
    const formData = new FormData();
    formData.append('file', file);

    // 3. Cinematic "Thinking" Delay (3 seconds)
    setTimeout(() => {
        fetch('/predict', { method: 'POST', body: formData })
        .then(res => res.json())
        .then(data => {
            if(data.error) { alert(data.error); resetApp(); return; }

            // Populate UI
            document.getElementById('res-img').src = document.getElementById('scan-img').src;
            document.getElementById('res-class').innerText = data.class.replace(/_/g, ' ');
            
            const conf = Math.round(data.confidence * 100);
            document.getElementById('res-conf').innerText = conf + "%";
            
            // Bar Animation
            setTimeout(() => document.getElementById('res-bar').style.width = conf + "%", 300);

            // Badge Color Logic
            const badge = document.getElementById('res-badge');
            if(data.class.toLowerCase().includes('healthy')) {
                badge.style.background = '#dcfce7';
                badge.style.color = '#166534';
                badge.innerText = "Healthy Crop";
            } else {
                badge.style.background = '#fee2e2';
                badge.style.color = '#991b1b';
                badge.innerText = "Disease Detected";
            }

            // Advice List
            const list = document.getElementById('res-advice');
            list.innerHTML = "";
            if(data.advice) {
                data.advice.forEach(tip => list.innerHTML += `<li>${tip}</li>`);
            } else {
                list.innerHTML = "<li>Consult a local expert.</li>";
            }

            // Switch View
            document.getElementById('view-scanner').style.display = 'none';
            document.getElementById('view-result').style.display = 'block';
        })
        .catch(err => {
            console.error(err);
            alert("Error connecting to server.");
            resetApp();
        });
    }, 3000); 
}

function resetApp() {
    document.getElementById('view-result').style.display = 'none';
    document.getElementById('view-scanner').style.display = 'none';
    document.getElementById('view-upload').style.display = 'block';
    fileInput.value = "";
    document.getElementById('res-bar').style.width = "0%";
}