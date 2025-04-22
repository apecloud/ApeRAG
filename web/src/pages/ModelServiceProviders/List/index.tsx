import { useIntl,history } from '@umijs/max';
import { useEffect } from 'react';
import './index.less';

interface IListProps {
  type: string;
  list: any;
}

const ModelServiceProviderList = (props: IListProps) => {
  const { type, list } = props;
  const classname = type === 'table' ? `data-list table` : 'data-list';

  const { formatMessage } = useIntl();
  const headerName = formatMessage({
    id: 'text.model_service_provider',
  });
  const uriName = formatMessage({
    id: 'text.model_service_provider_url',
  })
  const apiKeyName = formatMessage({
    id: 'text.model_service_provider_apikey',
  })

  return (
    <div className={`${classname}`}>
      <ul>
        <li className='header'>
            <div className="cell"><h5>{headerName}</h5></div>
            <div className="cell"><h5>{uriName}</h5></div>
            <div className="cell"><h5>{apiKeyName}</h5></div>
        </li>
        <li className='column'>
            <div className="cell"></div>
        </li>
        {list &&
          list.map((item: any) => {
            return (
              <li key={item.id} className='row' onClick={(e)=>{history.push(`/modelServiceProviders/${item.name}/settings`)}}>
                  <div className="cell title">
                    <h5>{item.label}</h5>
                  </div>
                  <div className="cell describe">
                    <p>{item.base_url}</p>
                  </div>
                  <div className="cell describe">
                    <p>{item.api_key}</p>
                  </div>
              </li>
            );
          })}
      </ul>
    </div>
  );
};

export default ModelServiceProviderList;
