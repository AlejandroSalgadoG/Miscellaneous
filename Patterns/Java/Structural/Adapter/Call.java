public class Call implements Comunication{

    private CellPhone cellPhone;

    Call(){
        cellPhone = new CellPhone();
    }

    public void comunicate(){
        cellPhone.makePhoneCall();
    }

}
