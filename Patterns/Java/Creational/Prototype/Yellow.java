public class Yellow extends Color{

    public Yellow(String info){
        this.info = info;
    }

    public String getInfo(){
        return info;
    }

    public Color clone(){
        return new Yellow(info);
    }

}
