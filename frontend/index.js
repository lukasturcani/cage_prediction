$(document).ready(function() {
    let nRows = 1;

    $('.add_row_button').click(function() {
        ++nRows;
        let newRow = emptyRow.clone(true, true)
        newRow.attr('id', `row${nRows}`)
        $(this).parent().before(newRow);
    });

    $('.predict_button').click(function() {
        let bb = 'O=Cc1cc(C=O)cc(C=O)c1';
        let lk = 'N[C@H]1CCCC[C@@H]1N';
        let model = 'amine2aldehyde3';
        let formData = new FormData();
        formData.append('bb', bb);
        formData.append('lk', lk);

        let request = new XMLHttpRequest();
        request.open('POST', `https://87d7e2cc.ngrok.io/predict/${model}`)
        request.addEventListener('load', function() {
            console.log(this.responseText);
        });
        request.send(formData);
    });

    let emptyRow = $('#row1').clone(true, true);
});
