import { Invitation, User } from '@/api';
import { PageContainer, PageHeader, RefreshButton } from '@/components';
import { DATETIME_FORMAT } from '@/constants';
import { api } from '@/services';
import { CheckOutlined, CloseOutlined, MoreOutlined } from '@ant-design/icons';
import { Button, Dropdown, Table, TableProps, Tabs, Typography } from 'antd';
import moment from 'moment';
import { useCallback, useEffect, useState } from 'react';

import { FormattedMessage, useIntl } from 'umi';

export default () => {
  const { formatMessage } = useIntl();
  const [users, setUsers] = useState<User[]>();
  const [usersLoading, setUsersLoading] = useState<boolean>();

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [invitations, setInvitations] = useState<Invitation[]>();
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [invitationsLoading, setInvitationsLoading] = useState<boolean>();

  const getUsers = useCallback(async () => {
    setUsersLoading(true);
    const res = await api.usersGet();
    setUsersLoading(false);
    setUsers(res.data.items);
  }, []);

  const getInvitations = useCallback(async () => {
    setInvitationsLoading(true);

    const res = await api.invitationsGet();
    setInvitationsLoading(false);
    setInvitations(res.data.items);
  }, []);

  const usersColumns: TableProps<User>['columns'] = [
    {
      title: formatMessage({ id: 'user.username' }),
      dataIndex: 'username',
    },
    {
      title: formatMessage({ id: 'user.email' }),
      dataIndex: 'email',
    },
    {
      title: formatMessage({ id: 'user.role' }),
      dataIndex: 'role',
    },
    {
      title: formatMessage({ id: 'user.status' }),
      dataIndex: 'is_active',
      render: (value) => {
        return value ? (
          <Typography.Text type="success">
            <CheckOutlined />
          </Typography.Text>
        ) : (
          <Typography.Text type="danger">
            <CloseOutlined />
          </Typography.Text>
        );
      },
    },
    {
      title: formatMessage({ id: 'text.createdAt' }),
      dataIndex: 'date_joined',
      width: 180,
      render: (value) => moment(value).format(DATETIME_FORMAT),
    },
    {
      title: formatMessage({ id: 'action.name' }),
      width: 80,
      render: () => {
        return (
          <Dropdown
            trigger={['click']}
            menu={{ items: [] }}
            overlayStyle={{ width: 160 }}
          >
            <Button type="text" icon={<MoreOutlined />} />
          </Dropdown>
        );
      },
    },
  ];

  useEffect(() => {
    getUsers();
    getInvitations();
  }, []);

  return (
    <PageContainer>
      <PageHeader
        title={formatMessage({ id: 'users.management' })}
        description={formatMessage({ id: 'users.management_tips' })}
      >
        <Button type="primary" disabled>
          <FormattedMessage id="users.invite" />
        </Button>
        <RefreshButton loading={usersLoading} onClick={() => getUsers()} />
      </PageHeader>

      <Tabs
        items={[
          {
            label: formatMessage({ id: 'users.all' }),
            key: 'users',
            children: (
              <Table
                rowKey="id"
                bordered
                columns={usersColumns}
                dataSource={users}
                loading={usersLoading}
              />
            ),
          },
          {
            label: formatMessage({ id: 'users.invitations' }),
            key: 'invitations',
            children: '',
            disabled: true,
          },
        ]}
      />
    </PageContainer>
  );
};
