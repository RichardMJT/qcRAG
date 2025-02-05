import uuid
from sqlalchemy import Column, Integer, String, DateTime, JSON, func, ForeignKey

from sqlalchemy.orm import relationship
from server.db.base import Base
from sqlalchemy.dialects.mysql import CHAR


class DetectedDataModel(Base):
    """仪器的诊断数据管理模型"""
    __tablename__ = 'detected_data'
    id = Column(CHAR(36), primary_key=True, default=lambda: str(uuid.uuid4()), comment='诊断数据ID')
    user_id = Column(CHAR(36), comment='用户ID')
    create_time = Column(DateTime, default=func.now(), comment='检测时间')
    detected_name = Column(String(100), comment='检测项目')
    # 记录检测过程中产生的中间数据
    operating_data = Column(JSON, default={})
    # 使用的仪器名称
    instrument = Column(String(30), comment='仪器名称')
    # 记录检测结果
    result = Column(String(30), comment='检测结果')
    # 记录经纬度
    location = Column(JSON, default={})
    # 记录所在城市
    city = Column(String(30), comment='城市')
    
    
    def __repr__(self):
        return f"""<DetectedDataModel(id={self.id},user_id={self.user_id}, detected_name={self.detected_name}) create_time={self.create_time}, 
        operating_data={self.operating_data}, result={self.result}, location={self.location}), 
        city={self.city})
        >"""