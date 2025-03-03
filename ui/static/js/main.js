// Main JavaScript for Financial Translation System

document.addEventListener('DOMContentLoaded', function() {
    // Auto-detect text direction for textareas
    const textareas = document.querySelectorAll('textarea');
    textareas.forEach(textarea => {
        textarea.addEventListener('input', function() {
            const text = this.value;
            const direction = isRTL(text) ? 'rtl' : 'ltr';
            this.setAttribute('dir', direction);
        });
        
        // Set initial direction
        if (textarea.value) {
            const direction = isRTL(textarea.value) ? 'rtl' : 'ltr';
            textarea.setAttribute('dir', direction);
        }
    });
    
    // Language switcher
    const sourceLangSelect = document.getElementById('source_language');
    const targetLangSelect = document.getElementById('target_language');
    
    if (sourceLangSelect && targetLangSelect) {
        sourceLangSelect.addEventListener('change', function() {
            // If source is changed to the same as target, swap them
            if (this.value === targetLangSelect.value) {
                targetLangSelect.value = this.value === 'en' ? 'ar' : 'en';
            }
        });
        
        targetLangSelect.addEventListener('change', function() {
            // If target is changed to the same as source, swap them
            if (this.value === sourceLangSelect.value) {
                sourceLangSelect.value = this.value === 'en' ? 'ar' : 'en';
            }
        });
    }
    
    // Swap languages button
    const swapButton = document.getElementById('swap_languages');
    if (swapButton) {
        swapButton.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Swap language selections
            const sourceValue = sourceLangSelect.value;
            sourceLangSelect.value = targetLangSelect.value;
            targetLangSelect.value = sourceValue;
            
            // Swap text content
            const sourceText = document.getElementById('source_text');
            const translationText = document.getElementById('translation');
            
            if (sourceText && translationText) {
                const tempText = sourceText.value;
                sourceText.value = translationText.value;
                translationText.value = tempText;
                
                // Update directions
                sourceText.setAttribute('dir', isRTL(sourceText.value) ? 'rtl' : 'ltr');
                translationText.setAttribute('dir', isRTL(translationText.value) ? 'rtl' : 'ltr');
            }
        });
    }
    
    // File upload preview
    const fileInput = document.getElementById('file_upload');
    const filePreview = document.getElementById('file_preview');
    
    if (fileInput && filePreview) {
        fileInput.addEventListener('change', function() {
            if (this.files && this.files[0]) {
                const file = this.files[0];
                filePreview.textContent = `Selected file: ${file.name} (${formatFileSize(file.size)})`;
            } else {
                filePreview.textContent = 'No file selected';
            }
        });
    }
    
    // Drag and drop for file upload
    const dropZone = document.querySelector('.file-upload-container');
    if (dropZone && fileInput) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, preventDefaults, false);
        });
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, unhighlight, false);
        });
        
        function highlight() {
            dropZone.classList.add('border-primary');
        }
        
        function unhighlight() {
            dropZone.classList.remove('border-primary');
        }
        
        dropZone.addEventListener('drop', handleDrop, false);
        
        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            
            fileInput.files = files;
            
            // Trigger change event
            const event = new Event('change', { bubbles: true });
            fileInput.dispatchEvent(event);
        }
    }
});

// Helper function to detect RTL text
function isRTL(text) {
    const rtlChars = /[\u0591-\u07FF\u200F\u202B\u202E\uFB1D-\uFDFD\uFE70-\uFEFC]/;
    return rtlChars.test(text);
}

// Helper function to format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
} 