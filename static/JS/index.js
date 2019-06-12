(function(){
    $(".Correct_Button").hide("slow")
    $(".Wrong_Button").hide("slow")
    $("#input_wrong").hide("slow")
    $(".Submit_Button").hide("slow")

    var canvas = document.querySelector( "#canvas" );
    var context = canvas.getContext( "2d" );
    canvas.width = 420;
    canvas.height = 420;
    var Mouse = { x: 0, y: 0 };
    var lastMouse = { x: 0, y: 0 };
    context.fillStyle="black";
    context.fillRect(0,0,canvas.width,canvas.height);
    context.color = "white";
    context.lineWidth = 15;
    context.lineJoin = context.lineCap = 'round';
    debug();
    canvas.addEventListener( "mousemove", function( e ){
        lastMouse.x = Mouse.x;
        lastMouse.y = Mouse.y;
        Mouse.x = e.pageX - this.offsetLeft;
        Mouse.y = e.pageY - this.offsetTop;
        }, false );
    canvas.addEventListener( "mousedown", function( e ){
        canvas.addEventListener( "mousemove", onPaint, false );
    }, false );
    canvas.addEventListener( "mouseup", function(){
        canvas.removeEventListener( "mousemove", onPaint, false );
    }, false );
    var onPaint = function(){
        context.lineWidth = context.lineWidth;
        context.lineJoin = "round";
        context.lineCap = "round";
        context.strokeStyle = context.color;
        context.beginPath();
        context.moveTo( lastMouse.x, lastMouse.y );
        context.lineTo( Mouse.x, Mouse.y );
        context.closePath();
        context.stroke();
    };
function debug(){
    /* CLEAR BUTTON */
    var clearButton = $( "#clearButton" );
    clearButton.on( "click", function(event){
        event.preventDefault();
        $(".Correct_Button").hide("slow")
        $(".Wrong_Button").hide("slow")
        $("#input_wrong").hide("slow")
        $(".Submit_Button").hide("slow")
        $('#result').text("!!!Get your prediction here!!!");
        context.clearRect( 0, 0, 420, 420 );
        context.fillStyle="black";
        context.fillRect(0,0,canvas.width,canvas.height);
    });
    /* LINE WIDTH */
    var slider = document.getElementById("myRange");
    var output = document.getElementById("sliderValue");
    output.innerHTML = slider.value;
    slider.oninput = function(){
        output.innerHTML = this.value;
        context.lineWidth = $( this ).val();
    }
    $( "#lineWidth" ).change(function(){
        context.lineWidth = $( this ).val();
    });
    }
}());
////////////////////////////////////////////////////////////

