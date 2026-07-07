$(document).ready(function() {
    // Delete prediction handler
    $(document).on('click', '.delete-prediction', function() {
        const $btn = $(this);
        const predictionId = $btn.data('prediction-id');
        
        if (confirm('Are you sure you want to delete this prediction?')) {
            $btn.prop('disabled', true);
            
            $.ajax({
                url: `/prediction/${predictionId}/delete`,
                type: 'POST',
                success: function(response) {
                    if (response.success) {
                        // Remove the table row
                        $(`tr[data-prediction-id="${predictionId}"]`).fadeOut(400, function() {
                            $(this).remove();
                            
                            // If no more predictions, show the "no predictions" message
                            if ($('.table tbody tr').length === 0) {
                                $('.table-responsive').replaceWith(
                                    '<p class="text-center">No predictions yet. Make your first prediction!</p>'
                                );
                            }
                        });
                    } else {
                        alert('Error deleting prediction: ' + response.error);
                        $btn.prop('disabled', false);
                    }
                },
                error: function() {
                    alert('Error deleting prediction');
                    $btn.prop('disabled', false);
                }
            });
        }
    });

    // Show/hide loading overlay
    function toggleLoading(show) {
        $('.loading-overlay').toggle(show);
    }

    function buildPayload() {
        return {
            temperature: parseFloat($('#temperature').val()),
            humidity: parseFloat($('#humidity').val()),
            ph: parseFloat($('#ph').val()),
            rainfall: parseFloat($('#rainfall').val()),
            nitrogen: parseFloat($('#nitrogen').val()),
            phosphorus: parseFloat($('#phosphorus').val()),
            potassium: parseFloat($('#potassium').val()),
            // Support both `#crop` (index page) and `#crop_type` (dashboard modal)
            crop: $('#crop').length ? $('#crop').val() : $('#crop_type').val()
        };
    }

    $('#previewBtn').on('click', function() {
        const payload = buildPayload();
        $('#payloadPreview').text(JSON.stringify(payload, null, 2));
        var previewModal = new bootstrap.Modal(document.getElementById('previewModal'));
        previewModal.show();
    });

    $('#confirmSend').on('click', function() {
        // send the payload shown in preview
        const data = buildPayload();
        $('#previewModal').modal('hide');
        sendPrediction(data);
    });

    $('#predictionForm').on('submit', function(e) {
        e.preventDefault();
        const data = buildPayload();
        sendPrediction(data);
    });

    $('#fetchAIBtn').on('click', function() {
        var fetchModal = new bootstrap.Modal(document.getElementById('fetchAIModal'));
        fetchModal.show();
    });

    $('#runAIQuery').on('click', function() {
        const query = $('#aiQuery').val();
        const lat = parseFloat($('#aiLat').val());
        const lon = parseFloat($('#aiLon').val());
        const start = $('#aiStart').val();
        const end = $('#aiEnd').val();

        $('#aiSuggestions').html('<p>Running planner…</p>');

        $.ajax({
            url: '/fetch_via_chatgpt',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ query: query, lat: lat, lon: lon, start: start, end: end }),
            success: function(resp) {
                if (resp.success) {
                    if (resp.suggestions && resp.suggestions.length > 0) {
                        let html = '<div class="list-group">';
                        resp.suggestions.forEach(function(s, idx) {
                            html += '<div class="list-group-item">';
                            html += '<h6>' + (s.name || s.source) + '</h6>';
                            html += '<p>' + (s.description || '') + '</p>';
                            html += '<pre>' + JSON.stringify(s.params, null, 2) + '</pre>';
                            html += '<button class="btn btn-sm btn-primary fetch-suggestion" data-idx="'+idx+'">Fetch</button>';
                            html += '</div>';
                        });
                        html += '</div>';
                        $('#aiSuggestions').html(html);

                        // attach click handlers for fetch
                        $('.fetch-suggestion').on('click', function() {
                            const index = $(this).data('idx');
                            const suggestion = resp.suggestions[index];
                            // Currently we only support nasa_power via /fetch_nasa
                            if (suggestion.source === 'nasa_power') {
                                const params = suggestion.params || {};
                                $.ajax({
                                    url: '/fetch_nasa',
                                    type: 'POST',
                                    contentType: 'application/json',
                                    data: JSON.stringify(params),
                                    success: function(fetchResp) {
                                        if (fetchResp.success) {
                                            $('#fetchedData').show();
                                            $('#fetchedPreview').html('<pre>' + JSON.stringify(fetchResp.preview, null, 2) + '</pre>');
                                            $('#fetchedFileLink').attr('href', '/' + fetchResp.file);
                                            // close modal
                                            $('#fetchAIModal').modal('hide');
                                        } else {
                                            alert('Fetch error: ' + fetchResp.error);
                                        }
                                    },
                                    error: function() {
                                        alert('Fetch failed');
                                    }
                                });
                            } else {
                                alert('Suggestion source not supported in UI: ' + suggestion.source);
                            }
                        });

                    } else {
                        $('#aiSuggestions').html('<p>No suggestions returned.</p>');
                    }
                } else {
                    $('#aiSuggestions').html('<p>Error: ' + resp.error + '</p>');
                }
            },
            error: function() {
                $('#aiSuggestions').html('<p>Planner request failed</p>');
            }
        });
    });

    function interpretYield(value) {
        // Interpret yield value (these ranges are examples and should be adjusted based on your specific crop/region)
        if (value > 8000) {
            return "Exceptional yield potential (top 1% production)";
        } else if (value > 6000) {
            return "Excellent yield potential (top 10% production)";
        } else if (value > 4000) {
            return "Good yield potential (above average)";
        } else if (value > 2000) {
            return "Moderate yield potential (average)";
        } else if (value > 1000) {
            return "Below average yield potential";
        } else {
            return "Low yield potential - consider soil amendments or different crop selection";
        }
    }

    function sendPrediction(data) {
        toggleLoading(true);
        $.ajax({
            url: '/predict',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(data),
            success: function(response) {
                if (response.success) {
                    const yieldValue = response.prediction.toFixed(2);
                    const interpretation = interpretYield(response.prediction);
                    
                    // Animate the result card
                    $('#prediction').html(`
                        <div class="yield-value" style="opacity: 0">${yieldValue} kg/hectare</div>
                        <div class="yield-interpretation text-muted" style="opacity: 0; font-size: 1rem; margin-top: 10px;">
                            <i class="fas fa-info-circle me-2"></i>${interpretation}
                        </div>
                    `);
                    
                    $('#result').slideDown(400, function() {
                        $('.yield-value').animate({opacity: 1}, 400);
                        setTimeout(() => {
                            $('.yield-interpretation').animate({opacity: 1}, 400);
                        }, 200);
                    });
                } else {
                    alert('Error: ' + response.error);
                }
            },
            error: function() {
                alert('Error making prediction request');
            },
            complete: function() {
                toggleLoading(false);
            }
        });
    }

});
