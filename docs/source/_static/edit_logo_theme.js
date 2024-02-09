document.addEventListener('DOMContentLoaded', function () {
    var logoImage = document.querySelector('img[src$="seismutils_logo_color.png"]'); // Modifica il selettore se necessario
    if (logoImage) {
        var theme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
        switch (theme) {
            case 'dark':
                logoImage.src = '_static/logos/seismutils_logo_light.png'; // Percorso per il logo chiaro
                break;
            case 'light':
                logoImage.src = '_static/logos/seismutils_logo_dark.png'; // Percorso per il logo scuro
                break;
        }
    }
});