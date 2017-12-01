var new_name = document.forms["update_form"]["img_new_name"];

function validate_update_form(){
    var name = new_name.value != "";
    var btn = private_btn2.checked || public_btn2.checked;

    if (name || btn) return true;
    
    alert("Nothing to modify");
    return false;
}
