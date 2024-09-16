const { useState, useEffect } = React;

const compressImage = (file, quality = 0.5) => {
  return new Promise((resolve) => {
    const reader = new FileReader()
    reader.readAsDataURL(file)
    reader.onload = () => {
      const img = new Image()
      img.src = reader.result
      img.onload = () => {
        const canvas = document.createElement('canvas')
        canvas.width = img.width
        canvas.height = img.height
        const ctx = canvas.getContext('2d')
        ctx.fillStyle = 'white'
        ctx.fillRect(0, 0, canvas.width, canvas.height)
        ctx.drawImage(img, 0, 0, img.width, img.height)
        canvas.toBlob(
          (blob) => {
            const newFile = new File([blob], file.name, {
              type: 'image/jpeg',
              lastModified: Date.now(),
            })
            resolve(newFile)
          },
          'image/jpeg',
          quality
        )
      }
    }
  })
}


const Header = ({ onImageUpload, pct, onDownload }) => {
  return (
    <div className="header">
      <label htmlFor="upload" className="btn">upload 🎩</label>
      <input type="file" accept="image/jpg,image/jpeg,image/png" id="upload" onChange={onImageUpload} />
      {pct && <div className="btn" onClick={onDownload}>download ✨</div>}
      {pct && (
        <div className="compress-text">
          compressed the file size by 
          &nbsp;<span>{pct}%</span>
        </div>
      )}
    </div>
  )
}

const Preview = ({ beforeSrc, beforeSize, afterSrc, afterSize }) => {
  return (
    <div className="preview">
      {beforeSrc 
        ? (
        <>
          <div className="img-box">
            <div className="title">
              Before: {beforeSize}
            </div>
            <img className="preview-before" src={beforeSrc} alt="" />
          </div>
          <div className="img-box">
            <div className="title">
              After: {afterSize}
            </div>
            <img className="preview-after" src={afterSrc} alt="" />
          </div>
        </>
      )
      : (
        <div>please upload an image 🦖</div>
      )}
    </div>
  )
}


const initImgSrc = 'https://images.unsplash.com/photo-1677002424307-d103e17f4bd6?crop=entropy&cs=tinysrgb&fm=jpg&ixid=MnwzMjM4NDZ8MHwxfHJhbmRvbXx8fHx8fHx8fDE2Nzk1NDM3MjI&ixlib=rb-4.0.3&q=80'
const App = () => {
  const [imgSrc, setImgSrc] = useState('')
  const [imgName, setImgName] = useState('')
  const [resultSrc, setResultSrc] = useState('')
  const [compressPct, setCompressPct] = useState(null)
  const [imgSize, setImgSize] = useState('')
  const [resultSize, setResultSize] = useState('')
  
  const handleSizeWithUnit = (size) => {
    const sizeInKb = Math.round(size / 1024)
    const isMb = Math.floor(sizeInKb / 1024)
    return isMb 
      ? Number((sizeInKb / 1024).toFixed(1)) + 'mb' 
      : sizeInKb + 'kb'
  }
  
  const onImageUpload = async (e) => {
    // reset
    setImgSrc('')
    setResultSrc('')
    setCompressPct(null)
    
    // get original image
    const file = e.target.files[0]
    const uploadSrc = URL.createObjectURL(file)
    setImgSrc(uploadSrc)
    setImgSize(handleSizeWithUnit(file.size))
    setImgName(file.name.split('.').slice(0, -1).join('.'))
    
    // get compressed image
    const res = await compressImage(file)
    const resUrl = URL.createObjectURL(res)
    setResultSrc(resUrl)
    setResultSize(handleSizeWithUnit(res.size))
    
    const pct = Math.round(res.size / file.size * 100)
    setCompressPct(100 - pct)
  }
  
  const onDownload = () => {
    const a = document.createElement('a')
    a.download = `${imgName}_compressed.png`
    a.href = resultSrc
    a.click()
  }
  
  return (
    <>
      <h1>Image Compression Tool</h1>
      <Header 
        onImageUpload={onImageUpload} 
        pct={compressPct}
        onDownload={onDownload}
        />
      <Preview 
        beforeSrc={imgSrc} 
        beforeSize={imgSize}
        afterSrc={resultSrc}
        afterSize={resultSize}
        />
    </>
  )
}


ReactDOM.render(
  <App />,
  document.getElementById('root')
);
