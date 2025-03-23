let slideIndex = 0;
const slideTexts = [
  "Step 1: Upload your video file",
  "Step 2: Click on Detect",
  "Step 3: View the result"
];

// Function to move slides
function moveSlide(direction) {
  const slides = document.querySelectorAll(".slide");
  slideIndex += direction;

  // Loop around slides
  if (slideIndex < 0) {
    slideIndex = slides.length - 1;
  } else if (slideIndex >= slides.length) {
    slideIndex = 0;
  }

  // Move to the correct slide
  const slideWidth = slides[0].offsetWidth;
  document.querySelector(".slides").style.transform = `translateX(-${slideIndex * slideWidth}px)`;

  // Change the text
  const textContainer = document.querySelector(".slide-text");
  textContainer.textContent = slideTexts[slideIndex];
}
    function scrollToSection(id) {
        document.getElementById(id).scrollIntoView({ behavior: 'smooth' });
    }

    function previewFile() {
const file = document.getElementById('file-upload').files[0];
const preview = document.getElementById('file-preview');

if (file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        preview.src = e.target.result;
        preview.style.display = 'block';
    };
    reader.readAsDataURL(file);
} else {
    preview.style.display = 'none';
    preview.src = '';
}
}