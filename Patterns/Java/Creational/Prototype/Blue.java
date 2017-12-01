public class Blue extends Color{

    public Blue(String info){
        this.info = info;
    }

    public String getInfo(){
        return info;
    }

    public Color clone(){
        return new Blue(info);
    }

}
