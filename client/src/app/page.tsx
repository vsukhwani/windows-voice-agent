"use client";

import {
  AudioClientHelper,
  ErrorCard,
  FullScreenContainer,
  LoaderSpinner,
} from "@pipecat-ai/voice-ui-kit";

import { App } from "./app";

export default function Home() {
  return (
    <FullScreenContainer>
      <AudioClientHelper
        transportType="smallwebrtc"
        connectParams={{
          connectionUrl: "/api/offer",
        }}
      >
        {({ handleConnect, handleDisconnect, loading, error }) =>
          loading ? (
            <LoaderSpinner />
          ) : error ? (
            <ErrorCard error={error} />
          ) : (
            <App
              handleConnect={handleConnect}
              handleDisconnect={handleDisconnect}
              error={error}
            />
          )
        }
      </AudioClientHelper>
    </FullScreenContainer>
  );
}
