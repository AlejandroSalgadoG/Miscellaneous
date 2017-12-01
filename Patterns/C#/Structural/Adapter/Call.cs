public class Call : Comunication{

    private CellPhone cellPhone;

    public Call(){
        cellPhone = new CellPhone();
    }

    public void comunicate(){
        cellPhone.makePhoneCall();
    }

}
